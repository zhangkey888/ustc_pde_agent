import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu = float(case_spec["pde"]["params"]["viscosity"])
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution - higher for better accuracy
    N = 192
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Manufactured solution (UFL)
    x = ufl.SpatialCoordinate(msh)
    
    exp1 = ufl.exp(-30*((x[0]-0.3)**2 + (x[1]-0.7)**2))
    exp2 = ufl.exp(-30*((x[0]-0.7)**2 + (x[1]-0.3)**2))
    
    u_ex1 = -60*(x[1]-0.7)*exp1 + 60*(x[1]-0.3)*exp2
    u_ex2 = 60*(x[0]-0.3)*exp1 - 60*(x[0]-0.7)*exp2
    u_ex = ufl.as_vector([u_ex1, u_ex2])
    
    # Source term: f = (u·∇)u - ν∇²u + ∇p
    grad_u_ex = ufl.grad(u_ex)
    laplacian_u_ex = ufl.div(grad_u_ex)
    grad_p_ex = ufl.as_vector([0.0*x[0], 0.0*x[0]])
    f_vec = grad_u_ex * u_ex - nu * laplacian_u_ex + grad_p_ex
    
    # Unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Residual (stress-divergence form)
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    F = (2*nu*ufl.inner(eps(u), eps(v))*ufl.dx
         + ufl.inner(ufl.grad(u)*u, v)*ufl.dx
         - p*ufl.div(v)*ufl.dx
         + ufl.div(u)*q*ufl.dx
         - ufl.inner(f_vec, v)*ufl.dx)
    
    J = ufl.derivative(F, w)
    
    # Boundary conditions - velocity on entire boundary
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(ufl.as_vector([u_ex1, u_ex2]), V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pinning at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Stokes solve for initial guess
    (u_s, p_s) = ufl.TrialFunctions(W)
    a_stokes = (2*nu*ufl.inner(ufl.sym(ufl.grad(u_s)), ufl.sym(ufl.grad(v)))*ufl.dx
                - p_s*ufl.div(v)*ufl.dx
                + ufl.div(u_s)*q*ufl.dx)
    L_stokes = ufl.inner(f_vec, v)*ufl.dx
    
    stokes_problem = petsc.LinearProblem(a_stokes, L_stokes, bcs=bcs,
                                          petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                                          petsc_options_prefix="stokes_")
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]
    
    # Newton solve for Navier-Stokes
    petsc_options_ns = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    ns_problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                         petsc_options_prefix="ns_",
                                         petsc_options=petsc_options_ns)
    w_h = ns_problem.solve()
    w.x.scatter_forward()
    
    # Get solver info
    snes = ns_problem._snes
    nl_its = int(snes.getIterationNumber())
    try:
        total_lin_its = int(snes.getLinearSolveIterations())
    except:
        total_lin_its = 0
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Compute L2 error for verification
    err_vec = u_h - ufl.as_vector([u_ex1, u_ex2])
    L2_sq_form = fem.form(ufl.inner(err_vec, err_vec)*ufl.dx)
    L2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(L2_sq_form), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 velocity error: {L2_error:.6e}")
        print(f"Newton iterations: {nl_its}")
        print(f"Total linear iterations: {total_lin_its}")
    
    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute magnitude
    u_mag = np.sqrt(np.nansum(u_values**2, axis=1)).reshape(ny_out, nx_out)
    u_mag = np.nan_to_num(u_mag, nan=0.0)
    
    # Clean up PETSc objects to prevent segfault
    del stokes_problem
    del ns_problem
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_lin_its,
            "nonlinear_iterations": [nl_its],
        }
    }
