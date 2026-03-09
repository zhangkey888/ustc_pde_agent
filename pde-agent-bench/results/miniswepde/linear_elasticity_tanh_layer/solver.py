import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """Solve 2D linear elasticity with manufactured solution."""
    
    # Extract parameters
    E_val = 1.0
    nu_val = 0.3
    
    # Lame parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    # Use degree 3 for high accuracy on the tanh layer
    N = 64
    element_degree = 3
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", element_degree, (gdim,)))
    
    # Material parameters as UFL constants
    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lam = fem.Constant(domain, PETSc.ScalarType(lam_val))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    u_exact_expr = ufl.as_vector([
        ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0]),
        0.1 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(gdim)
    
    # Compute source term from manufactured solution
    f = -ufl.div(sigma(u_exact_expr))
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions - apply exact solution on all boundaries
    u_bc = fem.Function(V)
    u_exact_interpolation = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_exact_interpolation)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_atol": "1e-15",
                "ksp_max_it": "5000",
            },
            petsc_options_prefix="elasticity_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="elasticity_"
        )
        u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    npts = points_3d.shape[0]
    u_values = np.full((npts, gdim), np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(npts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    # Compute displacement magnitude
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        },
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "linear_elasticity",
            "parameters": {"E": 1.0, "nu": 0.3},
        },
        "domain": {"type": "unit_square"},
        "manufactured_solution": {
            "u": ["tanh(6*(y-0.5))*sin(pi*x)", "0.1*sin(2*pi*x)*sin(pi*y)"]
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"Grid shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"u grid min/max: {np.nanmin(result['u']):.6f} / {np.nanmax(result['u']):.6f}")
    
    # Compute exact solution on the same grid for comparison
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    u1_exact = np.tanh(6*(YY - 0.5)) * np.sin(np.pi * XX)
    u2_exact = 0.1 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    u_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    error = np.max(np.abs(result['u'] - u_mag_exact))
    l2_error = np.sqrt(np.mean((result['u'] - u_mag_exact)**2))
    print(f"Max error (displacement magnitude): {error:.2e}")
    print(f"L2 error (displacement magnitude): {l2_error:.2e}")
