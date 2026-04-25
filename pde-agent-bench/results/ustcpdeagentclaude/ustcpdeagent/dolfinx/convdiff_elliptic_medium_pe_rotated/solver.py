import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    eps = 0.05
    beta_vec = np.array([3.0, 1.0])
    
    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh
    N = 112
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = sin(2*pi*(x+y))*sin(pi*(x-y))
    u_exact = ufl.sin(2*ufl.pi*(x[0]+x[1])) * ufl.sin(ufl.pi*(x[0]-x[1]))
    
    beta = ufl.as_vector([3.0, 1.0])
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    
    # f = -eps * lap(u_exact) + beta . grad(u_exact)
    grad_ue = ufl.grad(u_exact)
    lap_ue = ufl.div(grad_ue)
    f = -eps_c * lap_ue + ufl.dot(beta, grad_ue)
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = beta_norm * h / (2.0 * eps_c)
    # tau for SUPG (standard formula)
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe) - 1.0/Pe)
    
    # Galerkin form
    a_gal = eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = f * v * ufl.dx
    
    # SUPG residual-based stabilization
    # R(u) = -eps*lap(u) + beta.grad(u) - f
    # For P2, lap term is nonzero
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * (-eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # BCs from exact solution
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Get iterations
    try:
        iters = problem.solver.getIterationNumber()
    except Exception:
        iters = 0
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    
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
    
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    
    # Compute error vs exact
    grid = case_spec["output"]["grid"]
    xs = np.linspace(0, 1, grid["nx"])
    ys = np.linspace(0, 1, grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*(XX+YY)) * np.sin(np.pi*(XX-YY))
    err = np.sqrt(np.mean((result["u"] - u_ex)**2))
    print(f"RMSE: {err:.3e}")
    print(f"Max err: {np.max(np.abs(result['u'] - u_ex)):.3e}")
    print(f"Iterations: {result['solver_info']['iterations']}")
