import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Extract parameters from case_spec
    pde_spec = case_spec.get("pde", {})
    k_val = float(pde_spec.get("wavenumber", 25.0))
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = int(output.get("nx", 50))
    ny_out = int(output.get("ny", 50))
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_min = float(domain_spec.get("x_min", 0.0))
    x_max = float(domain_spec.get("x_max", 1.0))
    y_min = float(domain_spec.get("y_min", 0.0))
    y_max = float(domain_spec.get("y_max", 1.0))
    
    # Manufactured solution: u = exp(4*x)*sin(pi*y)
    # -∇²u - k²u = f
    # ∇²u = (16)*exp(4x)*sin(πy) + (-π²)*exp(4x)*sin(πy) = (16 - π²)*exp(4x)*sin(πy)
    # f = -∇²u - k²u = -(16 - π²)*exp(4x)*sin(πy) - k²*exp(4x)*sin(πy)
    #   = (π² - 16 - k²)*exp(4x)*sin(πy)
    
    # Adaptive mesh refinement
    best_u_grid = None
    best_info = None
    prev_norm = None
    
    # For k=25, we need fine mesh. With degree 2, we can use moderate resolution.
    # Rule of thumb: ~10 points per wavelength with degree 1, fewer with degree 2
    # wavelength = 2*pi/k ≈ 0.25, so need ~40 elements per unit with degree 1
    # With degree 2, ~20 elements per unit might suffice, but boundary layer exp(4x) needs more
    
    # Try different configurations
    configs = [
        (64, 2),   # moderate mesh, degree 2
        (96, 2),   # finer mesh, degree 2
        (128, 2),  # even finer
        (160, 2),  # finest fallback
    ]
    
    for N, deg in configs:
        try:
            u_grid, info, l2_norm = _solve_helmholtz(
                N, deg, k_val, x_min, x_max, y_min, y_max, nx_out, ny_out
            )
            
            if prev_norm is not None:
                rel_change = abs(l2_norm - prev_norm) / (abs(l2_norm) + 1e-15)
                if rel_change < 1e-4:
                    # Converged
                    return {"u": u_grid, "solver_info": info}
            
            prev_norm = l2_norm
            best_u_grid = u_grid
            best_info = info
            
        except Exception as e:
            print(f"Failed with N={N}, deg={deg}: {e}")
            continue
    
    return {"u": best_u_grid, "solver_info": best_info}


def _solve_helmholtz(N, deg, k_val, x_min, x_max, y_min, y_max, nx_out, ny_out):
    """Solve Helmholtz on NxN mesh with given element degree."""
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution (UFL expression)
    u_exact_ufl = ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = (π² - 16 - k²) * exp(4x) * sin(πy)
    k2 = k_val * k_val
    f_expr = (ufl.pi**2 - 16.0 - k2) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = u_exact on ∂Ω
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solver selection: try direct solver first for robustness with indefinite problem
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix="helmholtz_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to GMRES with ILU
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix="helmholtz_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
            },
        )
        u_sol = problem.solve()
    
    # Compute L2 norm for convergence check
    error_form = fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    
    norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
    l2_norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
    
    print(f"  N={N}, deg={deg}: L2 error = {l2_error:.6e}, L2 norm = {l2_norm:.6e}")
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, x_min, x_max, y_min, y_max, nx_out, ny_out)
    
    info = {
        "mesh_resolution": N,
        "element_degree": deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1 if ksp_type == "preonly" else 0,
    }
    
    return u_grid, info, l2_norm


def _evaluate_on_grid(domain, u_sol, x_min, x_max, y_min, y_max, nx_out, ny_out):
    """Evaluate the FE solution on a uniform grid."""
    
    xs = np.linspace(x_min, x_max, nx_out)
    ys = np.linspace(y_min, y_max, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
    return u_grid


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "wavenumber": 25.0,
        },
        "domain": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
        "output": {
            "nx": 50,
            "ny": 50,
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"\nWall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Check against exact solution on the grid
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(4 * XX) * np.sin(np.pi * YY)
    
    err = np.abs(result['u'] - u_exact)
    max_err = np.nanmax(err)
    l2_err = np.sqrt(np.nanmean(err**2))
    
    print(f"Max pointwise error: {max_err:.6e}")
    print(f"RMS error on grid: {l2_err:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
