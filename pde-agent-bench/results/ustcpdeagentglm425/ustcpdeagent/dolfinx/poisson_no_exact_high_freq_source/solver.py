import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse output grid spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Push mesh resolution higher to utilize time budget for maximum accuracy
    mesh_res = 256
    elem_deg = 3

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # Boundary conditions: u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Source term: f = sin(12*pi*x)*sin(10*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(12 * ufl.pi * x[0]) * ufl.sin(10 * ufl.pi * x[1])

    # Coefficient kappa = 1.0
    kappa = fem.Constant(domain, ScalarType(1.0))

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()

    # Gather across processes
    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-12,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}
