import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    kappa = 1.0
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005  # refine from suggested 0.01
    n_steps = int(round((t_end - t0) / dt_val))

    N = 64
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Manufactured: u = exp(-t)*sin(pi*x)*sin(pi*y)
    # u_t = -exp(-t) sin sin
    # -k*lap(u) = k*2*pi^2 * exp(-t) sin sin
    # f = u_t - k*lap(u) = (-1 + 2*pi^2*k) * exp(-t) sin sin
    u_exact = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-1.0 + 2.0 * ufl.pi**2 * kappa) * ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])  # t=0 -> exp(0)=1
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))

    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array

    # Boundary: Dirichlet, u = u_exact (= 0 on boundary since sin(0)=sin(pi)=0)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    # u_exact = 0 on boundary always, but we set it via interpolation each step to be safe
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Backward Euler weak form:
    # (u - u_n)/dt * v + k*grad(u).grad(v) = f * v
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    u_sol = fem.Function(V)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="heat_"
    )

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        # boundary stays zero
        problem.solve()
        ksp = problem.solver
        total_iters += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array

    # Sample onto output grid
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

    u_vals = np.full(nx * ny, np.nan)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny, nx)

    u_init_vals = np.full(nx * ny, np.nan)
    if points_on_proc:
        vals = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals.flatten()
    u_init_grid = u_init_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(spec)
    elapsed = time.time() - t0
    u = result["u"]
    # Compare with exact at t=0.1
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.1
    u_ex = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    linf = np.max(np.abs(u - u_ex))
    print(f"Time: {elapsed:.2f}s, L2 rms error: {err:.3e}, Linf: {linf:.3e}")
    print(f"Solver info: {result['solver_info']}")
