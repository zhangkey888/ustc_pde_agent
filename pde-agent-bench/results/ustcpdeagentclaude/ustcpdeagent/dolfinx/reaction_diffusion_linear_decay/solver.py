import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    pde_params = case_spec.get("pde", {}).get("params", {}) if "pde" in case_spec else {}
    epsilon = float(pde_params.get("epsilon", case_spec.get("epsilon", 0.01)))
    alpha = float(pde_params.get("reaction_alpha", case_spec.get("reaction_alpha", 1.0)))

    # Time
    time_cfg = case_spec.get("pde", {}).get("time", case_spec.get("time", {}))
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg.get("t_end", 0.6))
    dt_val = 0.002
    scheme = time_cfg.get("scheme", "backward_euler")

    # Output grid
    grid_cfg = case_spec["output"]["grid"]
    nx_out = grid_cfg["nx"]
    ny_out = grid_cfg["ny"]
    bbox = grid_cfg["bbox"]

    # Mesh
    N = 48
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Exact solution via UFL
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    def u_exact_ufl(t_c):
        return ufl.exp(-t_c) * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # f = du/dt - eps*Lap(u) + alpha*u
    # du/dt = -exp(-t) cos(2pi x) sin(pi y) = -u
    # Lap(u) = u * (-(2pi)^2 - pi^2) = -5*pi^2 * u
    # so f = -u - eps*(-5pi^2 u) + alpha*u = (-1 + 5*eps*pi^2 + alpha)*u
    u_ex = u_exact_ufl(t_const)
    f_expr = (-1.0 + 5.0 * epsilon * ufl.pi**2 + alpha) * u_ex

    # Initial condition
    u_n = fem.Function(V)
    u_n_expr = fem.Expression(u_exact_ufl(fem.Constant(domain, PETSc.ScalarType(t0))),
                               V.element.interpolation_points)
    u_n.interpolate(u_n_expr)

    # BC function
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Time step
    n_steps = int(np.ceil((t_end - t0) / dt_val))
    dt_used = (t_end - t0) / n_steps
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_used))

    # Weak form: backward Euler
    # (u - u_n)/dt - eps*Lap(u) + alpha*u = f(t_{n+1})
    # => (u/dt) + eps*grad(u).grad(v) + alpha*u - (u_n/dt + f) = 0
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u / dt_c) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + alpha * u * v * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=500)

    u_sol = fem.Function(V)

    # Store initial on output grid
    def sample_on_grid(func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        cells = []
        pts_use = []
        idx = []
        for i in range(pts.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                pts_use.append(pts[i])
                cells.append(links[0])
                idx.append(i)
        vals = np.full(pts.shape[0], np.nan)
        if len(pts_use) > 0:
            got = func.eval(np.array(pts_use), np.array(cells, dtype=np.int32))
            vals[idx] = got.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    total_iters = 0
    t = t0
    for step in range(n_steps):
        t += dt_used
        t_const.value = t
        # Update BC
        u_bc.interpolate(bc_expr)

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

    u_grid = sample_on_grid(u_n)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(total_iters),
            "dt": dt_used,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "params": {"epsilon": 0.01, "reaction_alpha": 1.0},
            "time": {"t0": 0.0, "t_end": 0.6, "dt": 0.01, "scheme": "backward_euler"},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    # Exact
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.6) * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex) ** 2))
    print(f"Wall time: {elapsed:.2f}s, L2 err: {err:.3e}")
    print("Solver info:", result["solver_info"])
