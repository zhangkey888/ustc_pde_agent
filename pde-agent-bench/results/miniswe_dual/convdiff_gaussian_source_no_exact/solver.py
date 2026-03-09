import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Parse parameters from case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    
    epsilon = params.get("epsilon", 0.02)
    beta_vec = params.get("beta", [8.0, 3.0])
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Adaptive mesh refinement with convergence check
    resolutions = [64, 128, 192]
    element_degree = 1
    
    prev_norm = None
    u_grid_final = None
    solver_info_final = None
    
    for N in resolutions:
        u_grid, solver_info, l2_norm = _solve_at_resolution(
            N, element_degree, epsilon, beta_vec,
            x_range, y_range, nx_out, ny_out, case_spec
        )
        
        if prev_norm is not None:
            rel_change = abs(l2_norm - prev_norm) / (abs(l2_norm) + 1e-15)
            if rel_change < 0.01:
                # Converged
                return {"u": u_grid, "solver_info": solver_info}
        
        prev_norm = l2_norm
        u_grid_final = u_grid
        solver_info_final = solver_info
    
    return {"u": u_grid_final, "solver_info": solver_info_final}


def _solve_at_resolution(N, degree, epsilon, beta_vec, x_range, y_range,
                          nx_out, ny_out, case_spec):
    """Solve at a given mesh resolution and return grid values + info."""
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Source term: f = exp(-250*((x-0.3)^2 + (y-0.7)^2))
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Diffusion coefficient
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    # Standard Galerkin weak form:
    # eps * (grad(u), grad(v)) + (beta . grad(u), v) = (f, v)
    a_std = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    
    # SUPG stabilization parameter (tau) - optimal formula
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG modification of test function: v_supg = tau * (beta . grad(v))
    # For P1 elements, laplacian(u) = 0, so strong residual = beta . grad(u) - f
    # SUPG adds: tau * (beta . grad(u) - f) * (beta . grad(v))
    r_supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), r_supg_test) * ufl.dx
    L_supg = f_expr * r_supg_test * ufl.dx
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    # Boundary conditions: u = 0 on all boundaries (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Solve with GMRES + ILU (good for non-symmetric convection-diffusion)
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    try:
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options_prefix=f"convdiff_{N}_{degree}_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "2000",
                "ksp_gmres_restart": "100",
            }
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options_prefix=f"convdiff_lu_{N}_{degree}_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            }
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, x_range, y_range, nx_out, ny_out)
    
    # Compute L2 norm for convergence check
    l2_norm = np.sqrt(np.nansum(u_grid**2) / max(np.count_nonzero(~np.isnan(u_grid)), 1))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }
    
    return u_grid, solver_info, l2_norm


def _evaluate_on_grid(domain, u_func, x_range, y_range, nx, ny):
    """Evaluate solution on a uniform nx x ny grid."""
    
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    
    # Create grid points (3D for dolfinx)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((xx.size, 3))
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(len(points), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "params": {
                "epsilon": 0.02,
                "beta": [8.0, 3.0],
            },
            "bcs": {},
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        },
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u = result["u"]
    info = result["solver_info"]
    
    print(f"Solution shape: {u.shape}")
    print(f"Solution range: [{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
    print(f"L2 norm: {np.sqrt(np.nanmean(u**2)):.6f}")
    print(f"NaN count: {np.isnan(u).sum()}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solver info: {info}")
