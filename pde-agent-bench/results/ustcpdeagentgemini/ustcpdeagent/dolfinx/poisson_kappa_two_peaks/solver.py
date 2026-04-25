import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Resolution
    mesh_res = 128
    degree = 2

    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_res, mesh_res], 
        cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Define exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.exp(0.5 * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Kappa
    kappa = 1.0 + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
            15.0 * ufl.exp(-200.0 * ((x[0] - 0.75)**2 + (x[1] - 0.75)**2))

    # Source term f = -div(kappa * grad(u_exact))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    # BC
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-9
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Interpolate to uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid_flat = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()

    # Get iterations
    iters = problem.solver.getIterationNumber()
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": iters
    }

    return {
        "u": u_grid_flat.reshape(ny, nx),
        "solver_info": solver_info
    }
