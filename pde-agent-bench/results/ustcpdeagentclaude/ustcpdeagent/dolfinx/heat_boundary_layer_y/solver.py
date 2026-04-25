import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Output grid
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]

    # Time params
    t0 = 0.0
    t_end = 0.08
    dt_val = 0.0005  # refined
    n_steps = int(round((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Exact solution u = exp(-t)*exp(5y)*sin(pi x)
    # u_t = -u
    # Δu = (-pi^2 + 25) * u
    # f = u_t - κ Δu = -u - (25 - pi^2) u = -(26 - pi^2) u
    kappa = 1.0
    u_exact = ufl.exp(-t_const) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    f_expr = (-1.0 - kappa * (25.0 - ufl.pi**2)) * u_exact

    # Initial condition
    u_n = fem.Function(V)
    u_bc = fem.Function(V)

    def exact_np(t):
        return lambda X: np.exp(-t) * np.exp(5.0 * X[1]) * np.sin(np.pi * X[0])

    u_n.interpolate(exact_np(0.0))

    # Boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Variational form - backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = u * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # BC setup - update each step
    bc = fem.dirichletbc(u_bc, bdofs)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)

    u_sol = fem.Function(V)
    total_iters = 0

    t_current = 0.0
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        # Update BC
        u_bc.interpolate(exact_np(t_current))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    cells = []
    points_on_proc = []
    eval_idx = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_idx.append(i)

    u_grid = np.zeros(nx_out * ny_out)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[eval_idx] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)

    # Initial condition grid
    u_init_grid = np.exp(5.0 * YY) * np.sin(np.pi * XX)

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
