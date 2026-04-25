import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    kappa_val = 5.0
    t0 = 0.0
    t_end = 0.08
    dt_val = 0.002
    n_steps = int(round((t_end - t0) / dt_val))

    N = 96
    degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # u_exact = exp(-t)*sin(2*pi*x)*sin(pi*y)
    # du/dt = -u_exact
    # -kappa * Lap(u) = -kappa * exp(-t) * (-(4pi^2 + pi^2)) sin(2pi x) sin(pi y)
    #                 = kappa * 5*pi^2 * u_exact
    # f = du/dt - kappa*Lap(u) = (-1 + kappa*5*pi^2) * u_exact
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-1.0 + kappa * 5.0 * ufl.pi ** 2) * u_exact_expr

    # Initial condition
    u_n = fem.Function(V)
    u_init_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])  # at t=0
    u_n.interpolate(fem.Expression(u_init_ufl, V.element.interpolation_points))

    # Save initial on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = []
    pts_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)
    pts_on = np.array(pts_on)
    cells = np.array(cells, dtype=np.int32)

    def sample(func):
        vals = np.zeros(nx_out * ny_out)
        ev = func.eval(pts_on, cells).flatten()
        for k, i in enumerate(idx_map):
            vals[i] = ev[k]
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample(u_n)

    # Variational form, backward Euler:
    # (u - u_n)/dt - kappa*Lap(u) = f(t^{n+1})
    # => u/dt + kappa*grad(u).grad(v) = u_n/dt + f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u / dt_c) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    # Dirichlet BC = u_exact on boundary
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        # update BC
        u_bc.interpolate(u_bc_expr)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]

    u_grid = sample(u_sol)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    res = solve(case)
    dt = time.time() - t0
    print(f"Wall time: {dt:.2f}s")
    # Compute error
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.08) * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((res["u"] - u_exact)**2))
    maxerr = np.max(np.abs(res["u"] - u_exact))
    print(f"L2 err: {err:.3e}, max err: {maxerr:.3e}")
    print(f"iters: {res['solver_info']['iterations']}")
