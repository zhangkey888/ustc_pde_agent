import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 144
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    kappa = fem.Constant(domain, PETSc.ScalarType(0.8))
    t0 = 0.0
    t_end = 0.12
    dt_val = 0.002
    n_steps = int(round((t_end - t0) / dt_val))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    a = u * v * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt * f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # Save initial condition on output grid later
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = 0.0

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="heat_"
    )

    total_iters = 0
    u_sol = None
    for step in range(n_steps):
        u_sol = problem.solve()
        total_iters += problem.solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

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

    u_values = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny, nx)
    u_init_grid = np.zeros((ny, nx))

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t = time.time()
    res = solve(spec)
    print(f"Time: {time.time()-t:.2f}s")
    print(f"u shape: {res['u'].shape}, max: {res['u'].max():.6f}, min: {res['u'].min():.6f}")
    print(f"iters: {res['solver_info']['iterations']}, n_steps: {res['solver_info']['n_steps']}")
