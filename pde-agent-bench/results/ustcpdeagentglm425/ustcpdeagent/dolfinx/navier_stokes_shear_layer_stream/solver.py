import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import time

def solve(case_spec: dict) -> dict:
    wall_start = time.time()
    comm = MPI.COMM_WORLD
    
    nu = 0.18
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    N = 192
    
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
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    
    # Exact velocity (manufactured solution)
    u_ex1 = 6*(1 - ufl.tanh(6*(x[1]-0.5))**2)*ufl.sin(ufl.pi*x[0])
    u_ex2 = -ufl.pi*ufl.tanh(6*(x[1]-0.5))*ufl.cos(ufl.pi*x[0])
    u_exact = ufl.as_vector([u_ex1, u_ex2])
    
    p_exact = ufl.cos(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])
    
    # Source term f = (u_ex . grad)u_ex - nu * div(grad(u_ex)) + grad(p_ex)
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Variational form
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    # --- Boundary Conditions ---
    u_bc_func = fem.Function(V)
    u_bc_expr = ufl.as_vector([u_ex1, u_ex2])
    u_bc_func.interpolate(
        fem.Expression(u_bc_expr, V.element.interpolation_points)
    )
    
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pinning at origin corner with correct value p(0,0)=1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 1.0  # p(0,0) = cos(0)*cos(0) = 1.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # --- Step 1: Solve Stokes for initial guess ---
    (u_stokes, p_stokes) = ufl.TrialFunctions(W)
    a_stokes = (
        2.0 * nu * ufl.inner(eps(u_stokes), eps(v)) * ufl.dx
        - p_stokes * ufl.div(v) * ufl.dx
        + ufl.div(u_stokes) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    
    a_stokes_form = fem.form(a_stokes)
    L_stokes_form = fem.form(L_stokes)
    
    A_stokes = petsc.assemble_matrix(a_stokes_form, bcs=bcs)
    A_stokes.assemble()
    
    b_stokes = petsc.assemble_vector(L_stokes_form)
    petsc.apply_lifting(b_stokes, [a_stokes_form], bcs=[bcs])
    b_stokes.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b_stokes, bcs)
    
    stokes_solver = PETSc.KSP().create(comm)
    stokes_solver.setOperators(A_stokes)
    stokes_solver.setType(PETSc.KSP.Type.PREONLY)
    stokes_pc = stokes_solver.getPC()
    stokes_pc.setType(PETSc.PC.Type.LU)
    stokes_pc.setFactorSolverType("mumps")
    stokes_solver.setFromOptions()
    
    w_stokes = fem.Function(W)
    stokes_solver.solve(b_stokes, w_stokes.x.petsc_vec)
    w_stokes.x.scatter_forward()
    
    stokes_its = stokes_solver.getIterationNumber()
    
    stokes_solver.destroy()
    A_stokes.destroy()
    b_stokes.destroy()
    
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    del w_stokes
    
    # --- Step 2: Newton solve for Navier-Stokes ---
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "snes_stol": 1e-12,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_rtol": 1e-10,
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    snes = problem.solver
    nl_its = snes.getIterationNumber()
    lin_its_total = snes.getLinearSolveIterations()
    ksp = snes.getKSP()
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()
    rtol = ksp.getTolerances()[0]
    
    # --- Compute error for verification ---
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    
    V_err = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    u_ex_func = fem.Function(V_err)
    u_ex_func.interpolate(fem.Expression(u_exact, V_err.element.interpolation_points))
    
    error_u_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx)
    )
    error_u = np.sqrt(comm.allreduce(error_u_sq, op=MPI.SUM))
    
    norm_u_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_ex_func, u_ex_func) * ufl.dx)
    )
    norm_u = np.sqrt(comm.allreduce(norm_u_sq, op=MPI.SUM))
    
    rel_error_u = error_u / norm_u if norm_u > 0 else error_u
    
    # --- Sample on output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        magnitude = np.linalg.norm(vals, axis=1)
        u_values[eval_map] = magnitude
    
    if comm.size > 1:
        u_values_clean = np.where(np.isnan(u_values), 0.0, u_values)
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values_clean, u_values_global, op=MPI.SUM)
        valid_mask = np.where(np.isnan(u_values), 0.0, 1.0)
        valid_global = np.zeros_like(valid_mask)
        comm.Allreduce(valid_mask, valid_global, op=MPI.SUM)
        u_values = np.where(valid_global > 0, u_values_global, np.nan)
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": lin_its_total + stokes_its,
            "nonlinear_iterations": [int(nl_its)],
        }
    }
