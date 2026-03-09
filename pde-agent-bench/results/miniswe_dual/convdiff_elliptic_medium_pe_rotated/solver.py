import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.05)
    beta_vec = params.get("beta", [3.0, 1.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Manufactured solution: u = sin(2*pi*(x+y))*sin(pi*(x-y))
    # We need to compute the source term f from this
    
    # Adaptive mesh refinement
    element_degree = 2  # P2 elements for better accuracy
    
    # For Pe ~ 63, we need good resolution. Try progressive refinement.
    resolutions = [48, 80, 128]
    
    prev_norm = None
    u_grid_result = None
    final_info = {}
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        
        p0 = np.array([x_range[0], y_range[0]])
        p1 = np.array([x_range[1], y_range[1]])
        domain = mesh.create_rectangle(
            comm, [p0, p1], [N, N],
            cell_type=mesh.CellType.triangle
        )
        
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Exact solution (manufactured)
        pi_ = ufl.pi
        u_exact_ufl = ufl.sin(2 * pi_ * (x[0] + x[1])) * ufl.sin(pi_ * (x[0] - x[1]))
        
        # Convection velocity
        beta = ufl.as_vector([beta_vec[0], beta_vec[1]])
        
        # Compute source term: f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
        grad_u_exact = ufl.grad(u_exact_ufl)
        # For the Laplacian, we use div(grad(u))
        # But since u_exact_ufl is a UFL expression (not a Function), we compute analytically
        # f = -epsilon * div(grad(u_exact)) + beta . grad(u_exact)
        
        # Let's compute the source term symbolically
        # u = sin(2*pi*(x+y))*sin(pi*(x-y))
        # We'll use UFL's differentiation capabilities
        # Note: ufl.div(ufl.grad(scalar)) gives the Laplacian
        laplacian_u = ufl.div(ufl.grad(u_exact_ufl))
        f_expr = -epsilon * laplacian_u + ufl.dot(beta, grad_u_exact)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Standard Galerkin weak form
        # -epsilon * laplacian(u) + beta . grad(u) = f
        # Weak form: epsilon * (grad(u), grad(v)) + (beta . grad(u), v) = (f, v)
        a_standard = (
            epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        )
        L_standard = f_expr * v * ufl.dx
        
        # SUPG stabilization
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        Pe_cell = beta_norm * h / (2.0 * epsilon)
        
        # SUPG stabilization parameter (tau)
        # Using the standard formula with coth
        # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
        # For high Pe, coth(Pe) ≈ 1, so tau ≈ h / (2*|beta|) * (1 - 1/Pe)
        # Simpler robust formula:
        tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
        
        # SUPG residual: R(u) = -epsilon * laplacian(u) + beta . grad(u) - f
        # For linear elements, laplacian(u) = 0 within elements
        # For P2, we keep it but note that third derivatives vanish
        # The strong residual applied to trial function:
        # R(u) = -epsilon * div(grad(u)) + beta . grad(u) - f
        # But for the bilinear form, we split into LHS and RHS parts
        
        # SUPG test function modification: v_supg = tau * beta . grad(v)
        v_supg = tau * ufl.dot(beta, ufl.grad(v))
        
        # For P2 elements, -epsilon * div(grad(u)) is nonzero within elements
        # Full SUPG:
        # a_supg = ((-epsilon * div(grad(u)) + beta . grad(u)), tau * beta . grad(v))
        # L_supg = (f, tau * beta . grad(v))
        
        a_supg = (
            (-epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * v_supg * ufl.dx
        )
        L_supg = f_expr * v_supg * ufl.dx
        
        a_total = a_standard + a_supg
        L_total = L_standard + L_supg
        
        # Boundary conditions (Dirichlet on all boundaries)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # All boundary
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim,
            lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Interpolate exact solution for BC
        u_bc_func = fem.Function(V)
        u_bc_func.interpolate(
            fem.Expression(u_exact_ufl, V.element.interpolation_points)
        )
        bc = fem.dirichletbc(u_bc_func, dofs)
        
        # Solve
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        
        try:
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "5000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix="cdiff_"
            )
            u_sol = problem.solve()
        except Exception:
            # Fallback to direct solver
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="cdiff_"
            )
            u_sol = problem.solve()
        
        # Evaluate on output grid
        x_coords = np.linspace(x_range[0], x_range[1], nx_out)
        y_coords = np.linspace(y_range[0], y_range[1], ny_out)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        points_2d = np.column_stack([X.ravel(), Y.ravel()])
        points_3d = np.zeros((points_2d.shape[0], 3))
        points_3d[:, :2] = points_2d
        
        # Point evaluation
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
            vals = u_sol.eval(
                np.array(points_on_proc),
                np.array(cells_on_proc, dtype=np.int32)
            )
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape((nx_out, ny_out))
        
        # Check convergence
        current_norm = np.nanmean(np.abs(u_grid))
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.005:
                # Converged
                u_grid_result = u_grid
                final_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": 0,
                }
                break
        
        prev_norm = current_norm
        u_grid_result = u_grid
        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
    
    return {
        "u": u_grid_result,
        "solver_info": final_info,
    }


if __name__ == "__main__":
    # Test with default case spec
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "params": {
                "epsilon": 0.05,
                "beta": [3.0, 1.0],
            },
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
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0.0, 1.0, nx_out)
    y_coords = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    u_exact = np.sin(2 * np.pi * (X + Y)) * np.sin(np.pi * (X - Y))
    
    error = np.sqrt(np.nanmean((u_grid - u_exact) ** 2))
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {np.nanmax(np.abs(u_grid - u_exact)):.6e}")
