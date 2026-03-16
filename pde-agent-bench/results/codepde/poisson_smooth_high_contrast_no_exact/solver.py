import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # 2. Create mesh - use higher resolution for accuracy with high-contrast coefficient
    nx, ny = 128, 128
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # 5. Define diffusion coefficient: kappa = 1 + 50*exp(-200*(x-0.5)**2)
    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5)**2)
    
    # 6. Define source term: f = 1 + sin(2*pi*x)*cos(2*pi*y)
    pi = ufl.pi
    f = 1.0 + ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    # 7. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 8. Boundary conditions: u = g on boundary
    # Check for boundary condition specification
    bc_spec = pde_config.get("bcs", [])
    g_value = 0.0
    g_expr = None
    if bc_spec:
        for bc_item in bc_spec:
            if bc_item.get("type") == "dirichlet":
                g_val = bc_item.get("value", 0.0)
                g_expr_str = bc_item.get("expr", None)
                if g_expr_str is not None:
                    g_expr = g_expr_str
                elif isinstance(g_val, (int, float)):
                    g_value = float(g_val)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    if g_expr is not None:
        u_bc.interpolate(lambda x_arr: eval(g_expr, {"x": x_arr[0], "y": x_arr[1], "np": np, "pi": np.pi, "sin": np.sin, "cos": np.cos}))
    else:
        u_bc.interpolate(lambda x_arr: np.full_like(x_arr[0], g_value))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 9. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 10. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
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
            "iterations": int(iterations),
        }
    }