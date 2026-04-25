import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    kappa_val = 0.5
    t0 = 0.0
    t_end = 0.12
    dt_val = 0.001  # finer than suggested for accuracy
    nsteps = int(round((t_end - t0) / dt_val))

    N = 128
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Exact solution: u = exp(-10*t)*sin(pi*x)*sin(pi*y)
    def u_exact_ufl(tc):
        return ufl.exp(-10.0 * tc) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # f = du/dt - kappa * laplacian(u)
    # du/dt = -10 * u ; laplacian(u) = -2*pi^2 * u
    # f = (-10 + 2*pi^2*kappa) * u
    u_ex = u_exact_ufl(t_const)
    f_expr = (-10.0 + 2.0 * ufl.pi**2 * kappa_val) * u_ex

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)  # previous time step
    # Initial condition
    u0_expr = fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points,
    )
    u_n.interpolate(u0_expr)

    # Backward Euler: (u - u_n)/dt - kappa*laplace(u) = f
    a = u * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_expr * v * ufl.dx

    # Boundary: u = exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)

    u_h = fem.Function(V)
    total_iters = 0

    t_cur = t0
    for step in range(nsteps):
        t_cur += dt_val
        t_const.value = t_cur

        # update BC
        u_bc.interpolate(u_bc_expr)

        # assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_h.x.array[:]

    # Save initial condition for reporting
    u_init = fem.Function(V)
    u_init.interpolate(u0_expr)

    # Sample onto grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idxs = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idxs.append(i)

    u_grid = np.full((ny * nx,), np.nan)
    u_init_grid = np.full((ny * nx,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[idxs] = vals.flatten()
        vals0 = u_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_grid[idxs] = vals0.flatten()

    u_grid = u_grid.reshape(ny, nx)
    u_init_grid = u_init_grid.reshape(ny, nx)

    # verification: compare to exact
    u_exact_grid = np.exp(-10.0 * t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.linalg.norm(u_grid - u_exact_grid) / np.sqrt(u_grid.size)
    print(f"RMS error vs exact: {err:.3e}, total KSP iters: {total_iters}")

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": nsteps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": True},
    }
    t0 = time.time()
    res = solve(spec)
    print("Wall time:", time.time() - t0)
    print("Shape:", res["u"].shape)
