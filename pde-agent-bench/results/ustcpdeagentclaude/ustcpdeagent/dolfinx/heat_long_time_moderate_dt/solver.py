import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    kappa_val = 0.5
    t0 = 0.0
    t_end = 0.2
    dt_val = 0.01
    
    # Resolution
    N = 80
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Exact solution: u = exp(-2t)*sin(pi*x)*sin(pi*y)
    u_exact_expr = ufl.exp(-2.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # du/dt - kappa * laplacian(u) = f
    # du/dt = -2 * u_exact
    # laplacian = -2*pi^2 * u_exact
    # f = -2*u - kappa*(-2*pi^2)*u = (-2 + 2*kappa*pi^2) * u_exact
    f_expr = (-2.0 + 2.0 * kappa_val * ufl.pi**2) * u_exact_expr
    
    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])  # t=0
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))
    
    # BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form - Backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = u * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)
    
    u_sol = fem.Function(V)
    
    # Save initial for output
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0
    
    t = t0
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        # Update BC
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
    
    # Sample on output grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)]).T
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny, nx)
    
    # Also initial
    u_init_func = fem.Function(V)
    t_const.value = 0.0
    u_init_func.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))
    u_init_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals.flatten()
    u_init_grid = u_init_values.reshape(ny, nx)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": True},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    # Compute exact solution
    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-2.0 * 0.2) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((result["u"] - u_exact)**2))
    print(f"Time: {elapsed:.2f}s, RMSE: {err:.2e}")
    print(f"Max err: {np.max(np.abs(result['u'] - u_exact)):.2e}")
    print(f"Info: {result['solver_info']}")
