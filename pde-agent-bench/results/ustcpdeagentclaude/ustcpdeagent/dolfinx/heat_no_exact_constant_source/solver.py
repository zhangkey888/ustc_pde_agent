import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def _sample_on_grid(u_sol, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        values[eval_map] = vals.flatten()
    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0_wall = time.time()
    comm = MPI.COMM_WORLD

    # Defaults from task
    t0 = 0.0
    t_end = 0.1
    dt_suggested = 0.02
    kappa_val = 1.0
    f_val = 1.0

    # Params we control
    N = 128           # mesh resolution
    degree = 2        # P2 elements
    dt = 0.002        # finer time step than suggested
    n_steps = int(round((t_end - t0) / dt))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Dirichlet BCs: u = 0 on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)       # previous time
    u_n.x.array[:] = 0.0
    u_sol = fem.Function(V)     # current
    u_sol.x.array[:] = 0.0

    kappa = fem.Constant(msh, PETSc.ScalarType(kappa_val))
    f_const = fem.Constant(msh, PETSc.ScalarType(f_val))
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt))

    # Backward Euler: (u - u_n)/dt - div(kappa grad u) = f
    # => u*v + dt*kappa*grad(u).grad(v) dx = (u_n + dt*f)*v dx
    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_c * f_const) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=500)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += ksp.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
        t_cur += dt

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    u_grid = _sample_on_grid(u_sol, msh, nx, ny, bbox)

    # Initial condition grid (zero)
    u_initial = np.zeros_like(u_grid)

    wall = time.time() - t0_wall

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": int(total_iters),
        "dt": dt,
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
        "wall_time": wall,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t_start = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t_start
    u = res["u"]
    print(f"WALL_TIME: {elapsed:.4f}")
    print(f"u shape: {u.shape}")
    print(f"u min/max: {u.min():.6e} / {u.max():.6e}")
    print(f"u mean: {u.mean():.6e}")
    print(f"solver_info: {res['solver_info']}")

    # Analytical reference: for f=1, kappa=1, zero BC, zero IC on unit square:
    # u(x,y,t) = sum_{m,n odd} 16/(pi^4 * m*n*(m^2+n^2)) * (1 - exp(-pi^2*(m^2+n^2)*t))
    #            * sin(m*pi*x) * sin(n*pi*y)
    nx, ny = u.shape[1], u.shape[0]
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.1
    u_ref = np.zeros_like(u)
    for m in range(1, 60, 2):
        for n in range(1, 60, 2):
            coef = 16.0 / (np.pi**4 * m * n * (m*m + n*n))
            decay = 1.0 - np.exp(-np.pi**2 * (m*m + n*n) * t_end)
            u_ref += coef * decay * np.sin(m*np.pi*XX) * np.sin(n*np.pi*YY)

    err = np.sqrt(np.mean((u - u_ref)**2))
    print(f"L2_ERROR: {err:.6e}")
    print(f"u_ref min/max: {u_ref.min():.6e} / {u_ref.max():.6e}")
