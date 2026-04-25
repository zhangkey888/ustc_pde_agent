import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    kappa = 0.5
    t0 = 0.0
    t_end = 0.2
    dt_val = 0.005  # smaller than suggested for better accuracy
    
    # Mesh resolution
    N = 80
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Exact: u = exp(-2t)*cos(pi x)*cos(pi y)
    # du/dt = -2 u
    # -kappa * laplacian(u) = -kappa * (-2 pi^2) u = 2*kappa*pi^2 u
    # f = du/dt - kappa lap u = -2 u + 2*kappa*pi^2 u = (2*kappa*pi^2 - 2) u
    u_exact_expr = ufl.exp(-2 * t_const) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    f_expr = (2 * kappa * ufl.pi**2 - 2) * u_exact_expr
    
    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    u_init_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(u_init_expr)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    a = u * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_expr * v * ufl.dx
    
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
    
    # Capture initial condition for grid output
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
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
    points_on_proc = np.array(points_on_proc)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)
    
    def sample(func):
        vals = np.full(pts.shape[0], np.nan)
        v = func.eval(points_on_proc, cells_on_proc)
        vals[eval_map] = v.flatten()
        return vals.reshape(ny_out, nx_out)
    
    u_initial_grid = sample(u_n)
    
    # Time stepping
    t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    total_iter = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        # Update BC
        u_bc.interpolate(u_init_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iter += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
    
    u_grid = sample(u_n)
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iter,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
        "pde": {"time": {}},
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    
    # Error
    nx_out = 64
    ny_out = 64
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-2 * 0.2) * np.cos(np.pi * XX) * np.cos(np.pi * YY)
    err = np.sqrt(np.mean((res["u"] - u_ex)**2))
    max_err = np.max(np.abs(res["u"] - u_ex))
    print(f"Elapsed: {elapsed:.3f}s  RMS Err: {err:.3e}  MaxErr: {max_err:.3e}")
    print(f"Info: {res['solver_info']}")
