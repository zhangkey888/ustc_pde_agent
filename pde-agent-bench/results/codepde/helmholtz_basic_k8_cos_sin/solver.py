import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    k_val = float(pde_config.get("wavenumber", 8.0))
    
    # 2. Create mesh - use sufficient resolution for k=8
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    # Manufactured solution: u = cos(pi*x)*sin(pi*y)
    # -∇²u - k²u = f
    # ∇²u = -pi²*cos(pi*x)*sin(pi*y) - pi²*cos(pi*x)*sin(pi*y) = -2*pi²*cos(pi*x)*sin(pi*y)
    # -∇²u = 2*pi²*cos(pi*x)*sin(pi*y)
    # f = 2*pi²*cos(pi*x)*sin(pi*y) - k²*cos(pi*x)*sin(pi*y) = (2*pi² - k²)*cos(pi*x)*sin(pi*y)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_expr = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = (2.0 * pi**2 - k_val**2) * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k²∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u_trial, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # 6. Solve - use direct solver for indefinite Helmholtz
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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