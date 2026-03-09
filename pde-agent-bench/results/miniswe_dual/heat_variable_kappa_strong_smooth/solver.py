import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """Solve the heat equation with variable kappa using backward Euler."""
    
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    t_end = float(time_params.get("t_end", 0.1))
    dt_val = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    N = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(3 * pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    kappa = 1.0 + 0.8 * ufl.sin(2 * pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    
    dudt = -ufl.exp(-t_const) * ufl.sin(3 * pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa * grad_u_exact)
    f_expr = dudt - div_kappa_grad_u
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    a = (u * v / dt_c) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ufl = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    a_form = fem.form(a)
    L_form = fem.form(L_ufl)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create vector from function space, not from form
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    t = 0.0
    n_steps = int(round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        bc_expr_obj = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        bc_func.interpolate(bc_expr_obj)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "heat", "time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
                "coefficients": {"kappa": {"type": "expr", "expr": "1 + 0.8*sin(2*pi*x)*sin(2*pi*y)"}}},
        "domain": {"type": "unit_square", "dim": 2}
    }
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Target: <= 2.24e-03, PASS: {error <= 2.24e-3}")
