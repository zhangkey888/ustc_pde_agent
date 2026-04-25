import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 48
    degree = 2
    dt_val = 0.005
    t_end = 0.25
    eps = 0.01
    alpha = 1.0
    beta = 1.0

    pde = case_spec.get("pde", {})
    params = pde.get("params", {}) if isinstance(pde, dict) else {}
    eps = float(params.get("epsilon", eps))
    alpha = float(params.get("reaction_alpha", alpha))
    beta = float(params.get("reaction_beta", beta))
    tcfg = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(tcfg.get("t0", 0.0))
    t_end = float(tcfg.get("t_end", t_end))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))

    def u_exact_ufl(tc):
        return ufl.exp(-tc) * (0.15 * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1]))

    def R(u):
        return alpha*u - beta*u**3

    ue = u_exact_ufl(t_const)
    dudt = -ue
    lap = ufl.div(ufl.grad(ue))
    f_expr = dudt - eps*lap + R(ue)

    u_n = fem.Function(V)
    t0_const = fem.Constant(msh, PETSc.ScalarType(t0))
    ue0 = ufl.exp(-t0_const) * (0.15 * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1]))
    u_n.interpolate(fem.Expression(ue0, V.element.interpolation_points))

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))

    F = ((u - u_n)/dt_c)*v*ufl.dx + eps*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx \
        + R(u)*v*ufl.dx - f_expr*v*ufl.dx

    # BC: exact solution value on boundary
    u_bc = fem.Function(V)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    J = ufl.derivative(F, u)
    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]; ny_out = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []; pop = []; emap = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pop.append(pts[i]); cells.append(links[0]); emap.append(i)
    pop = np.array(pop); cells = np.array(cells, dtype=np.int32)

    def sample(func):
        out = np.full((nx_out*ny_out,), np.nan)
        vals = func.eval(pop, cells).flatten()
        out[emap] = vals
        return out.reshape(ny_out, nx_out)

    u_initial = sample(u_n)

    # BC function needs updating each step
    ue_expr = fem.Expression(ue, V.element.interpolation_points)

    n_steps = int(round((t_end - t0) / dt_val))
    nl_iters = []
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        u_bc.interpolate(ue_expr)
        u.x.array[:] = u_n.x.array[:]
        problem.solve()
        try:
            it = problem.solver.getIterationNumber()
        except Exception:
            it = 1
        nl_iters.append(int(it))
        u_n.x.array[:] = u.x.array[:]

    u_grid = sample(u_n)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": sum(nl_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nl_iters,
        },
    }
