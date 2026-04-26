import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid spec
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox
    
    # Manufactured solution: u = exp(x)*sin(pi*y)
    # Delta u = (1 - pi^2)*exp(x)*sin(pi*y)
    # sigma = -Delta u = (pi^2 - 1)*exp(x)*sin(pi*y)
    # f = Delta^2 u = (1 - pi^2)^2 * exp(x)*sin(pi*y)
    pi_val = np.pi
    c1 = 1.0 - pi_val**2       # coefficient for Delta u
    c2 = c1**2                  # coefficient for f = Delta^2 u
    
    # Mesh resolution and element degree - target accuracy 2.62e-04
    mesh_res = 160
    elem_deg = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    x_coord = ufl.SpatialCoordinate(domain)
    
    # Exact expressions in UFL
    u_exact_ufl = ufl.exp(x_coord[0]) * ufl.sin(pi_val * x_coord[1])
    sigma_exact_ufl = (pi_val**2 - 1.0) * ufl.exp(x_coord[0]) * ufl.sin(pi_val * x_coord[1])
    f_ufl = c2 * ufl.exp(x_coord[0]) * ufl.sin(pi_val * x_coord[1])
    
    # ---- Step 1: Solve -Delta sigma = f with sigma = sigma_bc on boundary ----
    sigma = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(sigma), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_ufl, v) * ufl.dx  # since -Delta sigma = f => (grad sigma, grad v) = (f, v) with sigma_bc
    
    # Boundary conditions for sigma
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs_sigma = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    sigma_bc_func = fem.Function(V)
    sigma_bc_func.interpolate(
        fem.Expression(sigma_exact_ufl, V.element.interpolation_points)
    )
    bc_sigma = fem.dirichletbc(sigma_bc_func, boundary_dofs_sigma)
    
    # Solve Step 1
    problem1 = petsc.LinearProblem(a1, L1, bcs=[bc_sigma],
                                    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
                                    petsc_options_prefix="biharm_sigma_")
    sigma_sol = problem1.solve()
    sigma_sol.x.scatter_forward()
    
    # Get iteration count for step 1
    ksp1 = problem1.solver
    its1 = ksp1.getIterationNumber()
    
    # ---- Step 2: Solve -Delta u = sigma with u = g on boundary ----
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L2 = ufl.inner(sigma_sol, w) * ufl.dx
    
    # Boundary conditions for u
    boundary_dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs_u)
    
    # Solve Step 2
    problem2 = petsc.LinearProblem(a2, L2, bcs=[bc_u],
                                    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
                                    petsc_options_prefix="biharm_u_")
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count for step 2
    ksp2 = problem2.solver
    its2 = ksp2.getIterationNumber()
    
    # ---- Compute L2 error for verification ----
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    
    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_sq = comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM)
    l2_error = np.sqrt(l2_error_sq)
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
    
    # ---- Sample solution onto output grid ----
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    # Points shape: (3, N) for dolfinx
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather on all ranks (simple approach for small grids)
    from mpi4py import MPI as MPI4PY
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI4PY.SUM)
    
    # Replace any remaining NaN with 0 (shouldn't happen for points inside domain)
    u_grid = np.nan_to_num(u_values_global, nan=0.0).reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": its1 + its2,
        "l2_error": l2_error,
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    # Quick test
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {
            "time": False
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"L2 error: {result['solver_info']['l2_error']:.6e}")
