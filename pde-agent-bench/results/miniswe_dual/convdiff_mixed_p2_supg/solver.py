import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None):
    """Solve convection-diffusion with SUPG stabilization."""
    
    # Parameters
    epsilon = 0.01
    beta_vec = [10.0, 4.0]
    
    # Parse case_spec if provided
    if case_spec is not None:
        pde = case_spec.get('pde', {})
        params = pde.get('parameters', {})
        epsilon = params.get('epsilon', epsilon)
        beta_vec = params.get('beta', beta_vec)
    
    # Output grid
    nx_out, ny_out = 50, 50
    
    # Mesh resolution and element degree - need high accuracy
    N = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -ε ∇²u + β·∇u = f
    # u = sin(πx)sin(2πy)
    # ∇u = (π cos(πx) sin(2πy), 2π sin(πx) cos(2πy))
    # ∇²u = -π² sin(πx) sin(2πy) - 4π² sin(πx) sin(2πy) = -5π² sin(πx) sin(2πy)
    # -ε(-5π²u) + β·∇u = 5επ²u + β·∇u
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Source term derived from manufactured solution
    f_expr = -eps_c * ufl.div(ufl.grad(u_exact_expr)) + ufl.dot(beta, ufl.grad(u_exact_expr))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # Peclet number per element
    Pe_h = beta_norm * h / (2.0 * eps_c)
    
    # SUPG parameter (optimal for linear elements, also works well for P2)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)
    
    # SUPG residual: strong form residual applied to trial function
    # R(u) = -ε∇²u + β·∇u - f
    # For linear trial functions, ∇²u = 0 on each element, but for P2 it's nonzero
    # We use the full residual for P2
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_f = f_expr
    
    # SUPG test function modification
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = a_std + ufl.inner(R_u, supg_test) * ufl.dx
    L_supg = L_std + ufl.inner(R_f, supg_test) * ufl.dx
    
    # Boundary conditions - u = g on ∂Ω
    # For manufactured solution sin(πx)sin(2πy), u=0 on all boundaries of [0,1]²
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    x_pts = np.linspace(0, 1, nx_out)
    y_pts = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }
    
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_pts = np.linspace(0, 1, nx_out)
    y_pts = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    u_exact = np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
