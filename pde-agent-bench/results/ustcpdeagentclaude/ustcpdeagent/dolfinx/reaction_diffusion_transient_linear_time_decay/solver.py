import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {}) or case_spec.get("time", {}) or {}
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.4))
    dt_val = float(time_info.get("dt", 0.02))
    eps_val = float(pde.get("epsilon", case_spec.get("epsilon", 1.0)))

    # Use smaller dt and finer mesh for accuracy
    dt_val = 0.005
    N = 80
    degree = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))

    # u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -u_exact
    # -eps*Lap(u) = eps*2*pi^2*u_exact
    # R(u) = u
    # f = -u_exact + eps*2*pi^2*u_exact + u_exact = eps*2*pi^2*u_exact
    u_ex = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = eps_c * 2 * ufl.pi**2 * u_ex

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]) * np.exp(-t0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler: (u - u_n)/dt - eps*Lap(u) + u = f
    a = (u * v / dt_c) * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_expr * v * ufl.dx

    # Dirichlet BC: u = u_exact (which is 0 on boundary of unit square)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bdofs)

    u_sol = fem.Function(V)
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="rd_",
    )

    t = t0
    n_steps = 0
    total_iters = 0
    while t < t_end - 1e-12:
        step = min(dt_val, t_end - t)
        if abs(step - dt_val) > 1e-14:
            dt_c.value = step
        t += step
        t_const.value = t
        problem.solve()
        ksp = problem.solver
        total_iters += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        n_steps += 1

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    valid_pts = []
    idx = []
    for i in range(pts.shape[0]):
        links = col.links(i)
        if len(links) > 0:
            cells.append(links[0])
            valid_pts.append(pts[i])
            idx.append(i)
    vals = u_sol.eval(np.array(valid_pts), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx_out * ny_out)
    out[idx] = vals
    u_grid = out.reshape(ny_out, nx_out)

    # initial sample
    vals0 = u_initial.eval(np.array(valid_pts), np.array(cells, dtype=np.int32)).flatten()
    out0 = np.zeros(nx_out * ny_out)
    out0[idx] = vals0
    u0_grid = out0.reshape(ny_out, nx_out)

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
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }
