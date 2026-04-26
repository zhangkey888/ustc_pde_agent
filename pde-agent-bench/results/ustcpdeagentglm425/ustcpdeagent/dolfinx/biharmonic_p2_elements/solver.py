import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve biharmonic equation Δ²u = f on [0,1]² using mixed Ciarlet-Raviart formulation.
    Two sequential Poisson solves: -Δw = f, -Δu = w with simply supported BCs (u=0, w=0).
    """
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution and element degree
    N = 192
    elem_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Exact solution: u = sin(2πx)sin(2πy)
    # Δ²u = 64π⁴ sin(2πx)sin(2πy) = f
    # w = -Δu = 8π² sin(2πx)sin(2πy)
    
    x = ufl.SpatialCoordinate(domain)
    pi2 = 2.0 * np.pi
    
    u_exact = ufl.sin(pi2 * x[0]) * ufl.sin(pi2 * x[1])
    f_expr = 64.0 * np.pi**4 * ufl.sin(pi2 * x[0]) * ufl.sin(pi2 * x[1])
    
    # Interpolate f onto a function
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # First Poisson solve: -Δw = f, w = 0 on boundary
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_func, v) * ufl.dx
    
    w_bc = fem.Function(V)
    w_bc.x.array[:] = 0.0
    bc1 = fem.dirichletbc(w_bc, boundary_dofs)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc1],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="biharm_w_"
    )
    w_sol = problem1.solve()
    w_sol.x.scatter_forward()
    
    iterations1 = problem1.solver.getIterationNumber()
    
    # Second Poisson solve: -Δu = w, u = 0 on boundary
    u_trial = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(phi)) * ufl.dx
    L2 = ufl.inner(w_sol, phi) * ufl.dx
    
    u_bc_val = fem.Function(V)
    u_bc_val.x.array[:] = 0.0
    bc2 = fem.dirichletbc(u_bc_val, boundary_dofs)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc2],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="biharm_u_"
    )
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    
    iterations2 = problem2.solver.getIterationNumber()
    total_iterations = iterations1 + iterations2
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Points for evaluation (3D for dolfinx)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, tdim)
    
    # Find colliding cells
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Evaluate at points
    u_values = np.zeros(nx_out * ny_out)
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            u_values[i] = u_sol.eval(points[i].reshape(1, 3), np.array([links[0]], dtype=np.int32))[0]
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_sq = fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx))
    error_l2 = np.sqrt(comm.allreduce(float(error_sq), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_l2:.6e}")
        print(f"Iterations: w={iterations1}, u={iterations2}, total={total_iterations}")
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
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
