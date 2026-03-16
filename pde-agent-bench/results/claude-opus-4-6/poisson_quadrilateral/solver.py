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
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh (quadrilateral)
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_ufl = ufl.exp(x[0]) * ufl.cos(2 * pi * x[1])
    kappa = fem.Constant(domain, ScalarType(2.0))
    
    # f = -div(kappa * grad(u_exact))
    # grad(u_exact) = (exp(x)*cos(2*pi*y), -2*pi*exp(x)*sin(2*pi*y))
    # div(grad(u_exact)) = exp(x)*cos(2*pi*y) - 4*pi^2*exp(x)*cos(2*pi*y)
    #                     = exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    # f = -kappa * div(grad(u_exact)) = -2 * exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    f_expr = -kappa * ufl.div(ufl.grad(u_exact_ufl))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Variational form
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.cos(2 * np.pi * x[1]))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}