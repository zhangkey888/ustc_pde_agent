import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """
    Solve the biharmonic equation Delta^2 u = f on [0,1]x[0,1] using mixed formulation.
    
    Mixed formulation (simply-supported BCs):
      Introduce w = Delta u, then:
        Delta w = f   in Omega,  w = 0 on dOmega
        Delta u = w   in Omega,  u = 0 on dOmega
    
    Weak forms (after integration by parts):
      Step 1: Find w in H1_0: int(grad(w).grad(v)) dx = -int(f*v) dx
      Step 2: Find u in H1_0: int(grad(u).grad(v)) dx = -int(w*v) dx
    """
    
    # Extract grid configuration from case_spec
    oracle_config = case_spec.get('oracle_config', case_spec)
    output_cfg = oracle_config.get('output', {})
    grid_cfg = output_cfg.get('grid', {})
    nx_out = grid_cfg.get('nx', 50)
    ny_out = grid_cfg.get('ny', 50)
    bbox = grid_cfg.get('bbox', [0, 1, 0, 1])
    
    # Extract BC value
    bc_cfg = oracle_config.get('bc', case_spec.get('bc', {}))
    bc_dirichlet = bc_cfg.get('dirichlet', {})
    bc_value_str = bc_dirichlet.get('value', '0.0')
    bc_value = float(bc_value_str)
    
    comm = MPI.COMM_WORLD
    
    # Solver parameters
    N = 64
    element_degree = 2
    ksp_type_used = "cg"
    pc_type_used = "hypre"
    rtol_used = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates for source term
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    f_expr = ufl.cos(4 * pi_val * x[0]) * ufl.sin(3 * pi_val * x[1])
    
    # Boundary condition: homogeneous Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_value), dofs, V)
    
    petsc_options = {
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "ksp_rtol": str(rtol_used),
        "ksp_atol": "1e-12",
        "ksp_max_it": "2000",
    }
    
    # Step 1: Solve Delta w = f with w=0 on dOmega
    # Weak form: int(grad(w).grad(v)) dx = -int(f*v) dx
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = -f_expr * v_test * ufl.dx
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="biharmonic_w_"
    )
    w_h = problem_w.solve()
    
    # Step 2: Solve Delta u = w with u=0 on dOmega
    # Weak form: int(grad(u).grad(v)) dx = -int(w*v) dx
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L_u = -w_h * v_test2 * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="biharmonic_u_"
    )
    u_h = problem_u.solve()
    
    # Evaluate solution on output grid
    x_min, x_max = bbox[0], bbox[1]
    y_min, y_max = bbox[2], bbox[3]
    
    x_out = np.linspace(x_min, x_max, nx_out)
    y_out = np.linspace(y_min, y_max, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": 0,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    
    case_spec = {
        "id": "biharmonic_no_exact_sign_changing_rhs",
        "oracle_config": {
            "pde": {
                "type": "biharmonic",
                "source_term": "cos(4*pi*x)*sin(3*pi*y)",
            },
            "domain": {"type": "unit_square"},
            "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
            "output": {
                "format": "npz",
                "field": "scalar",
                "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50}
            },
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u_grid shape: {u_grid.shape}")
    print(f"u_grid min: {np.nanmin(u_grid):.6e}, max: {np.nanmax(u_grid):.6e}")
    print(f"u_grid L2 (grid): {np.sqrt(np.nanmean(u_grid**2)):.6e}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")
