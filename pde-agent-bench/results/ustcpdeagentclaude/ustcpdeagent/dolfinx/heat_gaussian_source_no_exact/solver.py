import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 176
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025
    n_steps = int(round((t_end - t0) / dt_val))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-200.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Crank-Nicolson
    a = u * v * ufl.dx + 0.5 * dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v - 0.5 * dt * kappa * ufl.inner(ufl.grad(u_n), ufl.grad(v)) + dt * f * v) * ufl.dx

    u_sol = fem.Function(V)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="heat_",
    )

    total_iters = 0
    for step in range(n_steps):
        problem.solve()
        try:
            total_iters += problem.solver.getIterationNumber()
        except Exception:
            pass
        u_n.x.array[:] = u_sol.x.array[:]

    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny, nx)

    u0_vals = np.zeros(pts.shape[0])
    if points_on_proc:
        vals0 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_vals[eval_map] = vals0.flatten()
    u0_grid = u0_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        },
    }
