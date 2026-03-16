import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    coeffs = pde_config.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 10.0))
    
    # Mesh and element parameters
    nx, ny = 64, 64
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution: u_exact = sin(pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Source term: f = -div(kappa * grad(u_exact))
    # grad(u_exact) = (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # div(kappa*grad(u_exact)) = kappa * (-pi^2*sin(pi*x)*sin(2*pi*y) - 4*pi^2*sin(pi*x)*sin(2*pi*y))
    #                          = -kappa * 5*pi^2 * sin(pi*x)*sin(2*pi*y)
    # So f = kappa * 5 * pi^2 * sin(pi*x)*sin(2*pi*y)
    f = kappa_val * 5.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    a = kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions - u_exact = sin(pi*x)*sin(2*pi*y) = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
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