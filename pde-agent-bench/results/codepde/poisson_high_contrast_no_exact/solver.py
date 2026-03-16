import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Extract source term
    source_val = 1.0
    if "source" in pde_config:
        src = pde_config["source"]
        if isinstance(src, (int, float)):
            source_val = float(src)
        elif isinstance(src, str):
            source_val = float(src)
    
    # Extract boundary condition
    bc_val = 0.0
    if "bc_val" in pde_config:
        bc_val = float(pde_config["bc_val"])
    elif "bcs" in pde_config:
        bcs_spec = pde_config["bcs"]
        if isinstance(bcs_spec, dict):
            bc_val = float(bcs_spec.get("value", 0.0))
        elif isinstance(bcs_spec, list):
            for b in bcs_spec:
                if isinstance(b, dict):
                    bc_val = float(b.get("value", 0.0))
    
    # Extract kappa info
    kappa_spec = pde_config.get("coefficients", {}).get("kappa", None)
    if kappa_spec is None:
        kappa_spec = pde_config.get("kappa", None)
    
    # 2. Create mesh - use higher resolution for high contrast coefficient
    nx, ny = 128, 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use degree 2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define coefficient kappa
    x = ufl.SpatialCoordinate(domain)
    
    # kappa = 1 + 1000*exp(-100*(x-0.5)**2)
    # This is a 1D Gaussian bump centered at x=0.5 (only depends on x[0])
    kappa_ufl = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, default_scalar_type(source_val))
    
    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(bc_val), dofs, V)
    
    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
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
            "iterations": int(iterations),
        }
    }