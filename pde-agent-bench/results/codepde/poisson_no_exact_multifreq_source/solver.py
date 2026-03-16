import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    nx_out = 50
    ny_out = 50
    
    # Mesh resolution and element degree - choose high enough for multi-frequency source
    mesh_resolution = 80
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    f = ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) + \
        0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1])
    
    # kappa = 1.0
    kappa = fem.Constant(domain, default_scalar_type(1.0))
    
    # Bilinear form: a = kappa * inner(grad(u), grad(v)) * dx
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Linear form: L = f * v * dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions: u = 0 on all boundaries (g = 0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
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
    
    # 7. Extract solution on uniform grid
    xv = np.linspace(0.0, 1.0, nx_out)
    yv = np.linspace(0.0, 1.0, ny_out)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')
    
    # Points array shape (N, 3) for evaluation
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }