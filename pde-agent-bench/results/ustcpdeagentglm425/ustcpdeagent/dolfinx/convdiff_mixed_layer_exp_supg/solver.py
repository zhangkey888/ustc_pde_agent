import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    epsilon = 0.01
    beta_vec = [12.0, 0.0]
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    N = 320
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    pi = np.pi
    u_exact_ufl = ufl.exp(3*x[0]) * ufl.sin(pi*x[1])
    
    laplacian_u = (9 - pi**2) * ufl.exp(3*x[0]) * ufl.sin(pi*x[1])
    beta_grad_u = 36 * ufl.exp(3*x[0]) * ufl.sin(pi*x[1])
    f_val = -epsilon * laplacian_u + beta_grad_u
    
    g_val = u_exact_ufl
    
    h = ufl.CellDiameter(domain)
    beta_ufl = ufl.as_vector(beta_vec)
    beta_norm = ufl.sqrt(ufl.inner(beta_ufl, beta_ufl))
    tau = h / (2.0 * beta_norm)
    
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a += ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
    a += tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx
    
    L = f_val * v * ufl.dx
    L += tau * f_val * ufl.dot(beta_ufl, ufl.grad(v)) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_val, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type": "gmres", "pc_type": "ilu",
                                                  "ksp_rtol": 1e-10},
                                   petsc_options_prefix="convdiff_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    error_sq = fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx))
    l2_error = np.sqrt(MPI.COMM_WORLD.allreduce(error_sq, MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": False}
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    print(f"Output shape: {result['u'].shape}")
