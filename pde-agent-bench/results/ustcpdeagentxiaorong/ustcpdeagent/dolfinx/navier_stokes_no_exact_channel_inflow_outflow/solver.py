import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = case_spec["pde"]["viscosity"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    N = 256
    
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    degree_u = 2
    degree_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Boundary conditions
    # Inflow: x = xmin
    inflow_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], bbox[0]))
    inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([
        4.0 * x[1] * (1.0 - x[1]),
        np.zeros(x.shape[1])
    ]))
    bc_inflow = fem.dirichletbc(u_inflow, inflow_dofs, W.sub(0))
    
    # No-slip bottom: y = ymin
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], bbox[2]))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bot = fem.Function(V)
    u_bot.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(u_bot, bottom_dofs, W.sub(0))
    
    # No-slip top: y = ymax
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], bbox[3]))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.x.array[:] = 0.0
    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    
    bcs = [bc_inflow, bc_bottom, bc_top]
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    nu_const = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Step 1: Stokes solve for initial guess
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v_s, q_s) = ufl.TestFunctions(W)
    
    a_stokes = (
        nu_const * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - p_s * ufl.div(v_s) * ufl.dx
        + ufl.div(u_s) * q_s * ufl.dx
    )
    L_stokes = ufl.inner(f, v_s) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    
    # Step 2: Newton solve for Navier-Stokes
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    J_form = ufl.derivative(F, w)
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": "1e-10",
            "snes_atol": "1e-12",
            "snes_max_it": "50",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((len(pts), gdim), 0.0)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)
    
    return {
        "u": vel_mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [1],
        }
    }
