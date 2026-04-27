import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Extract parameters
    k_val = case_spec["pde"]["helmholtz"]["k"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Choose mesh resolution and element degree
    element_degree = 3
    mesh_resolution = 80
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = cos(pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = -laplacian(u) - k^2 * u
    # laplacian(cos(pi*x)*sin(pi*y)) = -2*pi^2*cos(pi*x)*sin(pi*y)
    # f = (2*pi^2 - k^2)*cos(pi*x)*sin(pi*y)
    k_const = fem.Constant(domain, ScalarType(k_val))
    f = (2.0 * pi**2 - k_const**2) * u_exact
    
    # Bilinear form: integral(grad(u).grad(v) - k^2*u*v) dx = integral(f*v) dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve using direct solver (LU) - Helmholtz is indefinite
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
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
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }
