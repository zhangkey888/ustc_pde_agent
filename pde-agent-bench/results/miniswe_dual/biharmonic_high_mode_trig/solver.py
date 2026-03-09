import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict = None) -> dict:
    """
    Solve the biharmonic equation Δ²u = f on [0,1]² with u = 0 on ∂Ω.
    
    Manufactured solution: u = sin(3πx)sin(2πy)
    
    Mixed formulation (two sequential Poisson solves):
      σ = -Δu  =>  -Δu = σ  with u = 0 on ∂Ω
      -Δσ = f  with σ = 0 on ∂Ω  (since σ_exact = 13π²sin(3πx)sin(2πy) = 0 on ∂Ω)
    
    f = Δ²u = (9π² + 4π²)² sin(3πx)sin(2πy) = 169π⁴ sin(3πx)sin(2πy)
    """
    
    comm = MPI.COMM_WORLD
    
    nx_out = 50
    ny_out = 50
    
    if case_spec is not None:
        output = case_spec.get('output', {})
        nx_out = output.get('nx', nx_out)
        ny_out = output.get('ny', ny_out)
    
    # Parameters - degree 2 with moderate mesh should be sufficient
    N = 48
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    # Source term: f = 169π⁴ sin(3πx)sin(2πy)
    f_expr = (9*pi_val**2 + 4*pi_val**2)**2 * ufl.sin(3*pi_val*x[0]) * ufl.sin(2*pi_val*x[1])
    
    # Boundary conditions: both u and σ are zero on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Step 1: Solve -Δσ = f with σ = 0 on ∂Ω
    sigma_trial = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(sigma_trial), ufl.grad(w)) * ufl.dx
    L1 = ufl.inner(f_expr, w) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
        petsc_options_prefix="poisson1_"
    )
    sigma_sol = problem1.solve()
    
    # Step 2: Solve -Δu = σ with u = 0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(sigma_sol, v) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
        petsc_options_prefix="poisson2_"
    )
    u_sol = problem2.solve()
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    nx, ny = u_grid.shape
    x_out = np.linspace(0, 1, nx)
    y_out = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(3 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    mask = ~np.isnan(u_grid)
    max_err = np.nanmax(np.abs(u_grid - u_exact))
    rms_err = np.sqrt(np.nanmean((u_grid[mask] - u_exact[mask])**2))
    
    print(f"Mesh: {info['mesh_resolution']}, Degree: {info['element_degree']}")
    print(f"Max error on grid: {max_err:.6e}")
    print(f"RMS error on grid: {rms_err:.6e}")
    print(f"Time: {elapsed:.3f}s")
