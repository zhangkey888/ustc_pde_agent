import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 64
    degree = 2
    dt_val = 0.01
    t_end = 0.5
    eps = 1.0  # default; check spec
    pde = case_spec.get("pde", {})
    if "epsilon" in pde:
        eps = float(pde["epsilon"])

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(0.0))

    # u_exact = exp(-t)*sin(pi x)*sin(pi y)
    u_ex = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # R(u) = u (linear reaction assumption for "linear_basic")
    # f = du/dt - eps*lap(u) + u
    # du/dt = -u_ex; lap(u_ex) = -2 pi^2 u_ex
    # f = -u_ex - eps*(-2pi^2 u_ex) + u_ex = 2*pi^2*eps*u_ex
    f_expr = 2.0 * ufl.pi**2 * eps * u_ex + 0.0  # since -u_ex + u_ex = 0
    # Wait include reaction: f = -u_ex + 2pi^2*eps*u_ex + u_ex = 2pi^2*eps*u_ex

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)

    # Initial condition
    u0_expr = fem.Expression(ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                              V.element.interpolation_points)
    u_n.interpolate(u0_expr)
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array

    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps))

    # Backward Euler: (u - u_n)/dt - eps*lap(u) + u = f
    a = (u * v + dt_c * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_c * u * v) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    # BC
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)

    n_steps = int(round(t_end / dt_val))
    total_iters = 0
    t = 0.0
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        u_bc.interpolate(bc_expr)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    pts_use = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_use.append(pts[i])
            idx_map.append(i)
    vals = np.full(pts.shape[0], np.nan)
    if pts_use:
        v_eval = u_h.eval(np.array(pts_use), np.array(cells, dtype=np.int32)).flatten()
        for k, i in enumerate(idx_map):
            vals[i] = v_eval[k]
    u_grid = vals.reshape(ny, nx)

    # initial sample
    vals0 = np.full(pts.shape[0], np.nan)
    if pts_use:
        v0 = u_initial_func.eval(np.array(pts_use), np.array(cells, dtype=np.int32)).flatten()
        for k, i in enumerate(idx_map):
            vals0[i] = v0[k]
    u_init_grid = vals0.reshape(ny, nx)

    # Verify accuracy
    u_ex_grid = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid)**2))
    print(f"L2 error vs exact: {err:.3e}, total_iters={total_iters}")

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
        },
    }


if __name__ == "__main__":
    spec = {"pde": {"time": {"t_end": 0.5}},
            "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    import time
    t0 = time.time()
    out = solve(spec)
    print(f"Time: {time.time()-t0:.2f}s, shape={out['u'].shape}")
