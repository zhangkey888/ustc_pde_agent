import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the biharmonic equation using mixed formulation."""
    
    comm = MPI.COMM_WORLD
    
    # Parameters - need high accuracy (error <= 4.85e-05) within 3.2s
    # Use degree 2 elements with moderate mesh
    N = 64
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function spaces for mixed formulation
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_expr = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = Δ²u = (1 - π²)² * exp(x) * sin(πy)
    f_expr = (1.0 - ufl.pi**2)**2 * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Mixed formulation: introduce σ = -Δu
    # System:
    #   -Δu - σ = 0   (with u = g on ∂Ω)
    #   -Δσ = -f       (with σ = -Δu = -Δg on ∂Ω)
    
    # Actually, let's use the simpler approach:
    # Step 1: Solve -Δσ = f with σ = Δu_exact on boundary
    # Step 2: Solve -Δu = σ with u = u_exact on boundary
    # But σ = -Δu, so -Δ(-Δu) = f => Δ²u = f
    
    # Actually the standard mixed formulation is:
    # σ = Δu (or σ = -Δu depending on sign convention)
    # Then Δσ = Δ²u = f
    
    # Let's use: σ = -Δu, so -Δu = σ and Δσ = -f (i.e., -Δσ = f)
    # Step 1: Solve -Δσ = f with BC σ = -Δu_exact on ∂Ω
    # Step 2: Solve -Δu = σ with BC u = u_exact on ∂Ω
    
    # Compute Δu_exact = (1 - π²) * exp(x) * sin(πy)
    laplacian_u_exact = (1.0 - ufl.pi**2) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    # So σ = -Δu_exact = -(1 - π²) * exp(x) * sin(πy) = (π² - 1) * exp(x) * sin(πy)
    sigma_bc_expr = -laplacian_u_exact
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Step 1: Solve -Δσ = f with σ = sigma_bc on ∂Ω
    sigma_bc_func = fem.Function(V)
    sigma_bc_fem_expr = fem.Expression(sigma_bc_expr, V.element.interpolation_points)
    sigma_bc_func.interpolate(sigma_bc_fem_expr)
    
    bc_sigma = fem.dirichletbc(sigma_bc_func, boundary_dofs)
    
    sigma_h = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(sigma_h), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_expr, v) * ufl.dx
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_sigma],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    sigma_sol = problem1.solve()
    
    # Step 2: Solve -Δu = σ with u = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    u_bc_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_fem_expr)
    
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    u_h = ufl.TrialFunction(V)
    v2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_h), ufl.grad(v2)) * ufl.dx
    L2 = ufl.inner(sigma_sol, v2) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,  # placeholder
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Target error: 4.85e-05")
    print(f"Target time: 3.199s")
