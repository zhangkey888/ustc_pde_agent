import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Parameters
    kappa = 2.0
    pi = np.pi
    
    # Mesh resolution and element degree
    nx, ny = 80, 80
    degree = 2
    
    # 2. Create mesh (quadrilateral as specified by case ID)
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(x)*cos(2*pi*y)
    u_exact_ufl = ufl.exp(x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Compute source term: f = -div(kappa * grad(u_exact))
    # grad(u_exact) = (exp(x)*cos(2*pi*y), -2*pi*exp(x)*sin(2*pi*y))
    # div(kappa * grad(u_exact)) = kappa * (exp(x)*cos(2*pi*y) + exp(x)*(-4*pi^2)*cos(2*pi*y))
    #                             = kappa * exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    # f = -kappa * exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    #   = kappa * exp(x)*cos(2*pi*y)*(4*pi^2 - 1)
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    
    a = kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # 6. Boundary conditions - u = g on all boundaries
    # g = exact solution on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.cos(2 * np.pi * x[1]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
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
            "iterations": iterations,
        }
    }