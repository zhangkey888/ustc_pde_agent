import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    coeffs = pde_spec.get("coefficients", {})
    
    kappa = float(coeffs.get("kappa", 1.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt = float(time_spec.get("dt", 0.01))
    scheme = time_spec.get("scheme", "backward_euler")
    
    element_degree = 2
    N = 48
    
    result = _solve_at_resolution(N, element_degree, kappa, t_end, dt, scheme)
    
    u_sol = result["u_func"]
    domain = result["domain"]
    n_steps = result["n_steps"]
    total_iters = result["iterations"]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    u_values = _evaluate_function(u_sol, domain, points_3d)
    u_grid = u_values.reshape((nx_out, ny_out))
    
    u0_grid = np.exp(-40.0 * ((XX - 0.5)**2 + (YY - 0.5)**2))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme,
    }
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }


def _solve_at_resolution(N, degree, kappa, t_end, dt, scheme):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_var = fem.Constant(domain, ScalarType(0.0))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    gauss = ufl.exp(-40.0 * r2)
    
    u_exact_ufl = ufl.exp(-t_var) * gauss
    
    f_expr = ufl.exp(-t_var) * gauss * (-1.0 + 160.0 * kappa - 6400.0 * kappa * r2)
    
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    t_val = 0.0
    t_var.value = t_val
    
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_c = fem.Constant(domain, ScalarType(actual_dt))
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_expr * v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    for step in range(n_steps):
        t_val += actual_dt
        t_var.value = t_val
        
        bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    solver.destroy()
    
    return {
        "u_func": u_h,
        "domain": domain,
        "n_steps": n_steps,
        "iterations": total_iterations,
    }


def _evaluate_function(u_func, domain, points_3d):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    n_points = points_3d.shape[0]
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler",
            },
            "coefficients": {
                "kappa": 1.0,
            },
        }
    }
    
    t0 = time.perf_counter()
    result = solve(case_spec)
    elapsed = time.perf_counter() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_final = 0.1
    u_exact = np.exp(-t_final) * np.exp(-40.0 * ((XX - 0.5)**2 + (YY - 0.5)**2))
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
