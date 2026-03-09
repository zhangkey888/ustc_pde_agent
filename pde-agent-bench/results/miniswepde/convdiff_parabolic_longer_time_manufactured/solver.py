import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.05)
    beta_vec = params.get("beta", [2.0, 1.0])
    
    time_spec = pde.get("time", {})
    t_end = time_spec.get("t_end", 0.2)
    dt = time_spec.get("dt", 0.02)
    scheme = time_spec.get("scheme", "backward_euler")
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])

    N = 64
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    pi_val = np.pi
    
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    
    # Source term from manufactured solution
    # u_exact = exp(-2t) * sin(pi*x) * sin(pi*y)
    grad_u_exact = ufl.as_vector([
        ufl.exp(-2.0 * t_const) * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.exp(-2.0 * t_const) * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    ])
    dudt_exact = -2.0 * ufl.exp(-2.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    laplacian_u_exact = -2.0 * ufl.pi**2 * ufl.exp(-2.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    f_expr = dudt_exact - eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    u_n.interpolate(lambda x_arr: np.sin(pi_val * x_arr[0]) * np.sin(pi_val * x_arr[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.sin(pi_val * x_arr[0]) * np.sin(pi_val * x_arr[1]))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    # Backward Euler weak form with SUPG
    a = (u * v / dt_const) * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    
    L = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    R_lhs = u / dt_const + ufl.dot(beta, ufl.grad(u))
    R_rhs = u_n / dt_const + f_expr
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a += supg_test * R_lhs * ufl.dx
    L += supg_test * R_rhs * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        u_bc.interpolate(lambda x_arr: np.exp(-2.0 * t_val) * np.sin(pi_val * x_arr[0]) * np.sin(pi_val * x_arr[1]))
    
    update_bc(dt)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    t = 0.0
    n_steps = int(round(t_end / dt))
    total_iterations = 0
    
    u_h.x.array[:] = u_n.x.array[:]
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        update_bc(t)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(x_range[0], x_range[1], nx_out)
    ys = np.linspace(y_range[0], y_range[1], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Initial condition on same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.05,
                "beta": [2.0, 1.0],
            },
            "time": {
                "t_end": 0.2,
                "dt": 0.02,
                "scheme": "backward_euler",
            }
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    u_grid = result["u"]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-2.0 * 0.2) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"L2 grid error: {error:.6e}")
    print(f"Max error: {np.max(np.abs(u_grid - u_exact)):.6e}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
