import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None) -> dict:
    """Solve convection-diffusion with SUPG stabilization on quadrilateral mesh."""
    
    comm = MPI.COMM_WORLD
    
    # Parse parameters from case_spec or use defaults
    if case_spec is not None:
        pde = case_spec.get('pde', {})
        params = pde.get('parameters', {})
        epsilon = params.get('epsilon', 0.005)
        beta_vec = params.get('beta', [18.0, 6.0])
        domain_spec = case_spec.get('domain', {})
        output_spec = case_spec.get('output', {})
        nx_out = output_spec.get('nx', 50)
        ny_out = output_spec.get('ny', 50)
    else:
        epsilon = 0.005
        beta_vec = [18.0, 6.0]
        nx_out = 50
        ny_out = 50
    
    target_N = 80
    element_degree = 2
    
    # Create quadrilateral mesh
    domain = mesh.create_unit_square(
        comm, target_N, target_N, 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space - Q2 on quads
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for BCs and source term
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Convection velocity
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    
    # Source term from manufactured solution: f = -eps*laplacian(u) + beta.grad(u)
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    f = -epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin weak form
    a_standard = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    )
    L_standard = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG parameter
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    tau = h / (2.0 * beta_norm + 1e-10) * (1.0 - 1.0 / (Pe_cell + 1e-10))
    
    # SUPG additional terms
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # Total bilinear form and RHS
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # Boundary conditions - u = 0 on boundary (exact solution vanishes there)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Solve
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get solver iterations
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    nx_eval = nx_out
    ny_eval = ny_out
    
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": target_N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(iterations),
        }
    }
    
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"  Wall time: {elapsed:.3f}s")
    print(f"  Grid shape: {result['u'].shape}")
    print(f"  Grid min/max: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")
    
    # Check against exact solution on grid
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    diff = result['u'] - u_exact
    max_err = np.nanmax(np.abs(diff))
    l2_grid_err = np.sqrt(np.nanmean(diff**2))
    print(f"  Max grid error: {max_err:.6e}")
    print(f"  L2 grid error: {l2_grid_err:.6e}")
