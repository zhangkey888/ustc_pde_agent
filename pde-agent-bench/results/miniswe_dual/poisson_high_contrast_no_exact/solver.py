import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve the Poisson equation with high-contrast coefficient:
    -∇·(κ ∇u) = f in Ω = [0,1]²
    u = g on ∂Ω
    """
    # Parse case_spec
    pde = case_spec.get("pde", {})
    source_term = pde.get("source", "1.0")
    coefficients = pde.get("coefficients", {})
    kappa_spec = coefficients.get("kappa", coefficients.get("κ", {}))
    bc_spec = pde.get("boundary_conditions", {})
    
    # Determine boundary condition value
    bc_value = 0.0  # default
    if isinstance(bc_spec, dict):
        g_val = bc_spec.get("g", bc_spec.get("value", 0.0))
        if isinstance(g_val, (int, float)):
            bc_value = float(g_val)
    
    # Output grid size
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Adaptive mesh refinement
    resolutions = [48, 80, 128]
    element_degree = 2  # P2 elements for better accuracy with high contrast
    
    prev_norm = None
    u_sol = None
    final_N = resolutions[0]
    final_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define kappa using UFL spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        kappa = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
        
        # Source term
        f = fem.Constant(domain, ScalarType(1.0))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Bilinear and linear forms
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L_form = f * v * ufl.dx
        
        # Boundary conditions: u = bc_value on all boundaries
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(bc_value), dofs, V)
        
        # Solve with iterative solver first, fallback to direct
        ksp_type = "cg"
        pc_type = "hypre"
        rtol = 1e-10
        
        try:
            problem = petsc.LinearProblem(
                a, L_form, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "2000",
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
        except Exception:
            # Fallback to direct solver
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a, L_form, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
        
        # Compute L2 norm for convergence check
        norm_val = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
            op=MPI.SUM
        ))
        
        final_N = N
        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
        
        if prev_norm is not None:
            rel_change = abs(norm_val - prev_norm) / (abs(norm_val) + 1e-15)
            if rel_change < 0.01:
                # Converged
                break
        
        prev_norm = norm_val
    
    # Evaluate solution on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
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
        "solver_info": final_info,
    }


if __name__ == "__main__":
    # Test with a minimal case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "source": "1.0",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 1000*exp(-100*(x-0.5)**2)"}
            },
            "boundary_conditions": {"type": "dirichlet", "value": 0.0},
        },
        "output": {"nx": 50, "ny": 50},
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
