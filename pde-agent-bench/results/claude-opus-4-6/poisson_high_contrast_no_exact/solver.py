import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 80
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Coefficient kappa = 1 + 1000*exp(-100*(x-0.5)**2)
    # Note: x-0.5 refers to x[0]-0.5 based on the expression
    kappa = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
    
    # Source term
    f = fem.Constant(domain, ScalarType(1.0))
    
    # Bilinear and linear forms
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω
    # Check if there's a specific g in case_spec
    bc_spec = case_spec.get("pde", {}).get("bcs", [])
    g_value = 0.0  # default
    for bc in bc_spec:
        if bc.get("type") == "dirichlet":
            g_expr = bc.get("expr", "0")
            if g_expr == "0" or g_expr == "0.0":
                g_value = 0.0
            else:
                g_value = float(g_expr) if isinstance(g_expr, (int, float, str)) else 0.0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(g_value), dofs, V)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on 50x50 grid
    n_grid = 50
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_grid * n_grid))
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
    
    u_values = np.full(n_grid * n_grid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_grid, n_grid))
    
    # Clean up PETSc objects
    ksp.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }