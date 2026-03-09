cat <<'EOF' > solver.py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _create_case_spec_default():
    case_spec = {
        "case_id": "heat_gaussian_source_no_exact",
        "pde": {
            "type": "parabolic",
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler",
            },
        },
        "coefficients": {
            "kappa": 1.0,
        },
        "domain": {
            "type": "unit_square",
        },
    }
    return case_spec


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    time_info = case_spec.get("pde", {}).get("time", {})
    t_end = float(time_info.get("t_end", 0.1))
    dt = float(time_info.get("dt", 0.02))
    time_scheme = time_info.get("scheme", "backward_euler")

    mesh_resolution = int(case_spec.get("mesh_resolution", 64))
    element_degree = int(case_spec.get("element_degree", 1))

    domain = mesh.create_unit_square(
        comm,
        mesh_resolution,
        mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
    bc = fem.dirichletbc(u_bc, dofs)

    u_n = fem.Function(V)

    def u0_expr(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    u_n.interpolate(u0_expr)

    x = ufl.SpatialCoordinate(domain)
    f_ufl = ufl.exp(-200.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f_expr)

    kappa = float(case_spec.get("coefficients", {}).get("kappa", 1.0))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (1.0 / dt) * ufl.inner(u, v) * ufl.dx + kappa_c * ufl.inner(
        ufl.grad(u), ufl.grad(v)
    ) * ufl.dx
    L = (1.0 / dt) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_func, v) * ufl.dx

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
    rtol = 1e-8
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    u_sol = fem.Function(V)

    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0

    nx = ny = 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    def sample_to_grid(u_func):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(
            domain, cell_candidates, points.T
        )

        pts_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                pts_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)

        values = np.full(points.shape[1], np.nan, dtype=np.float64)
        if pts_on_proc:
            pts_arr = np.array(pts_on_proc, dtype=np.float64)
            cells_arr = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts_arr, cells_arr).flatten()
            values[eval_map] = vals

        return values.reshape((nx, ny))

    u_initial_grid = sample_to_grid(u_n)

    for _ in range(n_steps):
        t += dt

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    u_grid = sample_to_grid(u_sol)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp.getType().lower(),
        "pc_type": pc.getType().lower() if pc.getType() is not None else "none",
        "rtol": rtol,
        "iterations": int(total_iterations),
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
    }

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
    return result


if __name__ == "__main__":
    cs = _create_case_spec_default()
    out = solve(cs)
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", out["u"].shape)
        print("u_initial shape:", out["u_initial"].shape)
        print("solver_info:", out["solver_info"])
EOF

python solver.py
