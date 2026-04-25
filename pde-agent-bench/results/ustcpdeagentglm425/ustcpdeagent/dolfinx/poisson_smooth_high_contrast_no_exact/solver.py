import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD

def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 256
    elem_deg = 2
    rtol = 1e-12
    ksp_type = "cg"
    pc_type = "hypre"

    domain = dmesh.create_unit_square(comm, mesh_res, mesh_res)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    fdim = domain.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5)**2)
    f = 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    t1 = time.perf_counter()
    wall_time = t1 - t0

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

    u_grid = np.zeros((nx_out * ny_out,), dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()

    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global

    u_grid = u_grid.reshape(ny_out, nx_out)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }
    return result
