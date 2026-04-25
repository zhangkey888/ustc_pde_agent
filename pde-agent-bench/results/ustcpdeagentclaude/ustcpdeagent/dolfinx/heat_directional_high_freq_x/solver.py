import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Time parameters
    t0 = 0.0
    t_end = 0.08
    dt_val = 0.001  # smaller than suggested for better accuracy
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    # Exact solution expression
    def u_exact_expr(t_val):
        return ufl.exp(-t_val) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Source: u_t - kappa*Laplacian(u) = f
    # u_t = -exp(-t)*sin(8pi x)*sin(pi y)
    # Laplacian(u) = -(64pi^2 + pi^2) exp(-t) sin(8pi x) sin(pi y)
    # f = u_t - Lap(u) = -exp(-t)sin(8pi x)sin(pi y) + (65 pi^2) exp(-t) sin(8pi x) sin(pi y)
    #   = (65 pi^2 - 1) exp(-t) sin(8pi x) sin(pi y)
    kappa = 1.0
    f_expr = (-ufl.exp(-t_const) * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
              + kappa * (64 + 1) * ufl.pi**2 * ufl.exp(-t_const) * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]))

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
                                  V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    u_initial_field = u_n.copy()

    # Dirichlet BC - exact solution on boundary
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_expr(t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational problem: Backward Euler
    # (u - u_n)/dt - kappa * Lap(u) = f
    # => u/dt + kappa*grad(u).grad(v) = u_n/dt + f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = (u / dt_c) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=1000)

    uh = fem.Function(V)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        u_bc.interpolate(bc_expr)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_grid[idx] = vals[k, 0] if vals.ndim > 1 else vals[k]

    u_grid = u_grid.reshape(ny_out, nx_out)

    # Initial
    u_init_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_initial_field.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_init_grid[idx] = vals0[k, 0] if vals0.ndim > 1 else vals0[k]
    u_init_grid = u_init_grid.reshape(ny_out, nx_out)

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
    case_spec = {
        "pde": {"time": True},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
    }
    t0 = time.time()
    out = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.2f}s")

    # Verify
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.08
    u_exact = np.exp(-t_end) * np.sin(8*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((out["u"] - u_exact)**2))
    print(f"L2 error: {err:.4e}")
    print(f"Max error: {np.max(np.abs(out['u'] - u_exact)):.4e}")
    print(f"Iterations: {out['solver_info']['iterations']}")
