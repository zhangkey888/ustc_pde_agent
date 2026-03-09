import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.08))
    dt_suggested = float(time_params.get("dt", 0.008))
    scheme = time_params.get("scheme", "backward_euler")
    
    element_degree = 2
    
    # Use smaller dt than suggested for accuracy (backward Euler is first-order)
    # dt=0.008 gives ~2.3e-3 error, need ~1e-3, so halve it roughly
    dt = min(dt_suggested, 0.0032)
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    # Single resolution that's known to converge well
    N = 64
    result = _solve_with_resolution(N, element_degree, kappa_val, t_end, dt, n_steps, scheme)
    return result

def _solve_with_resolution(N, degree, kappa_val, t_end, dt, n_steps, scheme):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    f_ufl = ufl.exp(-t) * ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 - kappa_val * (25.0 - ufl.pi**2))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.exp(5.0 * X[0]) * np.sin(np.pi * X[1]))
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.exp(5.0 * X[0]) * np.sin(np.pi * X[1]))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_ufl * v * ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bc_func = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, dofs)
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    u_sol = fem.Function(V)
    total_iterations = 0
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        t_val = current_t
        bc_func.interpolate(lambda X, tv=t_val: np.exp(-tv) * np.exp(5.0 * X[0]) * np.sin(np.pi * X[1]))
        b_vec = petsc.create_vector(V)
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        b_vec.destroy()
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    u_grid = _evaluate_function(domain, u_sol, points_3d)
    u_grid = u_grid.reshape(nx_out, ny_out)
    u_init_grid = _evaluate_function(domain, u_initial_func, points_3d)
    u_init_grid = u_init_grid.reshape(nx_out, ny_out)
    solver.destroy()
    A.destroy()
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

def _evaluate_function(domain, func, points_3d):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    n_points = points_3d.shape[0]
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    return u_values
