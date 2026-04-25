
import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "```DIAGNOSIS\nequation_type: convection_diffusion\nspatial_dim: 2\ndomain_geometry: rectangle\nunknowns: scalar\ncoupling: none\nlinearity: linear\ntime_dependence: transient\nstiffness: stiff\ndominant_physics: mixed\npeclet_or_reynolds: high\nsolution_regularity: smooth\nbc_type: all_dirichlet\nspecial_notes: manufactured_solution\n```"

METHOD = "```METHOD\nspatial_method: fem\nelement_or_basis: Lagrange_P1\nstabilization: supg\ntime_method: backward_euler\nnonlinear_solver: none\nlinear_solver: gmres\npreconditioner: ilu\nspecial_treatment: none\npde_skill: convection_diffusion / reaction_diffusion / biharmonic\n```"


def build_case_defaults(case_spec):
    if case_spec is None:
        case_spec = {}
    pde = case_spec.setdefault("pde", {})
    pde.setdefault("t0", 0.0)
    pde.setdefault("t_end", 0.2)
    pde.setdefault("dt", 0.02)
    pde.setdefault("scheme", "backward_euler")
    pde.setdefault("epsilon", 0.05)
    pde.setdefault("beta", [2.0, 1.0])
    pde.setdefault("time", True)
    output = case_spec.setdefault("output", {})
    output.setdefault("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    return case_spec


def exact_ufl(domain, t_value):
    x = ufl.SpatialCoordinate(domain)
    return ufl.exp(-2.0 * t_value) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def source_ufl(domain, eps, beta, t_value):
    x = ufl.SpatialCoordinate(domain)
    uex = ufl.exp(-2.0 * t_value) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ut = -2.0 * uex
    grad_u = ufl.grad(uex)
    lap_u = ufl.div(grad_u)
    beta_vec = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))
    return ut - eps * lap_u + ufl.dot(beta_vec, grad_u)


def sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_values[np.array(eval_ids, dtype=np.int32)] = vals
    gathered = domain.comm.gather(local_values, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution at all requested grid points")
        return merged.reshape((ny, nx))
    return None


def initial_grid_from_exact(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.sin(np.pi * XX) * np.sin(np.pi * YY)


def run_case(case_spec, mesh_resolution, degree, dt, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    case_spec = build_case_defaults(case_spec)
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    eps = float(case_spec["pde"]["epsilon"])
    beta = np.array(case_spec["pde"]["beta"], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta))
    t0 = float(case_spec["pde"]["t0"])
    t_end = float(case_spec["pde"]["t_end"])
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)

    u_n.interpolate(fem.Expression(exact_ufl(domain, t0), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc.interpolate(fem.Expression(exact_ufl(domain, t0 + dt), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    beta_vec = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))
    f_expr = source_ufl(domain, eps, beta, t0 + dt)

    h = ufl.CellDiameter(domain)
    pk = beta_norm * h / (2.0 * eps + 1.0e-14)
    tau = h / (2.0 * beta_norm + 1.0e-14) * ((ufl.cosh(pk)/ufl.sinh(pk)) - 1.0 / pk)

    a_std = (u * v / dt + eps * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L_std = (u_n * v / dt + f_expr * v) * ufl.dx
    strong_u = u / dt - eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    strong_rhs = u_n / dt + f_expr
    a = a_std + tau * ufl.dot(beta_vec, ufl.grad(v)) * strong_u * ufl.dx
    L = L_std + tau * ufl.dot(beta_vec, ufl.grad(v)) * strong_rhs * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    total_iterations = 0
    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        u_bc.interpolate(fem.Expression(exact_ufl(domain, t_now), V.element.interpolation_points))
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        try:
            solver.solve(b, u_h.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(exact_ufl(domain, t_end), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = u_h.x.array - u_exact.x.array
    e.x.scatter_forward()
    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(e * e * ufl.dx)), op=MPI.SUM))

    grid = case_spec["output"]["grid"]
    u_grid = sample_on_grid(domain, u_h, grid)
    if comm.rank == 0:
        return {
            "u": u_grid,
            "u_initial": initial_grid_from_exact(grid),
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": str(solver.getType()),
                "pc_type": str(solver.getPC().getType()),
                "rtol": float(rtol),
                "iterations": int(total_iterations),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": "backward_euler",
                "l2_error_fem": float(l2_error),
            },
        }
    return {"u": None, "u_initial": None, "solver_info": {}}


def solve(case_spec: dict) -> dict:
    case_spec = build_case_defaults(case_spec)
    return run_case(case_spec, mesh_resolution=40, degree=1, dt=0.01, ksp_type="gmres", pc_type="ilu", rtol=1e-8)


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "t0": 0.0,
            "t_end": 0.2,
            "dt": 0.02,
            "scheme": "backward_euler",
            "epsilon": 0.05,
            "beta": [2.0, 1.0],
            "time": True,
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    tic = time.perf_counter()
    result = solve(case_spec)
    toc = time.perf_counter()
    if MPI.COMM_WORLD.rank == 0:
        grid = case_spec["output"]["grid"]
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_ex = np.exp(-2.0 * case_spec["pde"]["t_end"]) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
        l2_grid = math.sqrt(np.mean((result["u"] - u_ex) ** 2))
        print(f"L2_ERROR: {l2_grid:.12e}")
        print(f"WALL_TIME: {toc - tic:.12e}")
        print(result["u"].shape)
        print(result["solver_info"])
