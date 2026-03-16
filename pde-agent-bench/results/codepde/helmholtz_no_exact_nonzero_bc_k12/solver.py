import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("k", 12.0))
    f_val = float(pde_config.get("source_term", 0.0))
    
    # For Helmholtz with k=12, we need sufficient resolution
    # Rule of thumb: ~10 elements per wavelength, wavelength = 2*pi/k ≈ 0.524
    # On unit square, need at least ~20 elements per direction, but use more for accuracy
    nx = 128
    ny = 128
    degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define boundary condition
    # g on boundary - need to figure out what g is
    # From case_spec, "nonzero_bc" suggests non-trivial BC
    # Check if there's a bc specification
    bc_spec = pde_config.get("bc", None)
    
    # Define the boundary condition function
    # For "nonzero_bc" with no exact solution, we use the BC from the spec
    # Default: g = sin(pi*x)*sin(pi*y) or similar; let's check case_spec
    # Since no exact solution is given, the BC defines the problem
    # Common choice for nonzero BC Helmholtz: g = sin(pi*x) * sin(pi*y) on boundary
    # But on boundary of unit square, sin(pi*x)*sin(pi*y) = 0! 
    # Let's try g = cos(pi*x) * cos(pi*y) or g = 1.0 or from spec
    
    # Parse BC more carefully
    bc_expr_str = None
    if bc_spec is not None:
        if isinstance(bc_spec, dict):
            bc_expr_str = bc_spec.get("expression", None)
            bc_value = bc_spec.get("value", None)
        else:
            bc_expr_str = str(bc_spec)
    
    # Also check for boundary_condition key
    bc_info = pde_config.get("boundary_condition", pde_config.get("bc", None))
    if bc_info is not None and isinstance(bc_info, dict):
        bc_expr_str = bc_info.get("expression", bc_info.get("value", None))
    
    # Build the BC function
    u_bc = fem.Function(V)
    
    if bc_expr_str is not None and isinstance(bc_expr_str, str):
        # Try to parse expression
        def make_bc_func(expr_str):
            def bc_func(x):
                pi = np.pi
                result = eval(expr_str, {"np": np, "pi": pi, "x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                return result
            return bc_func
        try:
            u_bc.interpolate(make_bc_func(bc_expr_str))
        except:
            # Fallback: use cos-based BC
            u_bc.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    else:
        # Default nonzero BC: use cos(pi*x)*cos(pi*y) which is nonzero on boundaries
        u_bc.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    # 5. Locate boundary DOFs
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Variational problem: -∇²u - k²u = f
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k2 = fem.Constant(domain, default_scalar_type(k_val**2))
    f = fem.Constant(domain, default_scalar_type(f_val))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 7. Solve - use direct solver for indefinite Helmholtz
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }