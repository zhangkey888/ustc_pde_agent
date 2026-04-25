import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

DIAGNOSIS = "convection_diffusion transient scalar high_Peclet all_dirichlet variable_coeff"
METHOD = "fem Lagrange_P1 SUPG backward_euler gmres ilu"

def _get_case(case_spec: dict):
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    eps = float(pde.get("epsilon", case_spec.get("epsilon", 0.02)))
    beta = np.array(pde.get("beta", case_spec.get("beta", [6.0, 3.0])), dtype=np.float64)
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.02)))
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return eps, beta, t0, t_end, dt, nx, ny, bbox

def _forcing_at_time(t):
    def f(x):
        return np.exp(-150.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2)) * np.exp(-t)
    return f

def _initial_condition(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

def _sample_function(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals_local[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(vals_local, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return None

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    eps, beta_np, t0, t_end, dt_suggested, nx_out, ny_out, bbox = _get_case(case_spec)

    mesh_resolution = max(144, int(case_spec.get("agent_params", {}).get("mesh_resolution", 144)))
    target_dt = min(dt_suggested, 0.0025)
    n_steps = max(1, int(math.ceil((t_end - t0) / target_dt)))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt_suggested
    degree = int(case_spec.get("agent_params", {}).get("element_degree", 1))
    ksp_type = str(case_spec.get("agent_params", {}).get("ksp_type", "gmres"))
    pc_type = str(case_spec.get("agent_params", {}).get("pc_type", "ilu"))
    rtol = float(case_spec.get("agent_params", {}).get("rtol", 1e-8))

    wall0 = time.perf_counter()

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta = fem.Constant(domain, np.array(beta_np, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(eps))
    dt_c = fem.Constant(domain, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.interpolate(_initial_condition)
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]

    f = fem.Function(V)
    f.interpolate(_forcing_at_time(t0 + dt))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-14)
    Pe = beta_norm * h / (2.0 * eps_c + 1.0e-14)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-14)
    tau_supg = h / (2.0 * beta_norm) * (cothPe - 1.0 / (Pe + 1.0e-14))

    residual_u = (u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_rhs = (u_n / dt_c) + f
    streamline_test = ufl.dot(beta, ufl.grad(v))

    a = ((u / dt_c) * v + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    a += tau_supg * residual_u * streamline_test * ufl.dx
    L = ((u_n / dt_c) + f) * v * ufl.dx + tau_supg * residual_rhs * streamline_test * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)

    uh = fem.Function(V)
    total_iterations = 0

    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        f.interpolate(_forcing_at_time(t))
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    accuracy_check = None
    mass = fem.assemble_scalar(fem.form(uh * ufl.dx))
    l2_sq = fem.assemble_scalar(fem.form(uh * uh * ufl.dx))
    if rank == 0:
        accuracy_check = {"solution_mass": float(mass), "solution_l2_norm": float(np.sqrt(abs(l2_sq)))}

    u_grid = _sample_function(domain, uh, nx_out, ny_out, bbox)
    u0_grid = _sample_function(domain, u_initial, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": float(dt),
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    if rank == 0:
        solver_info.update(accuracy_check)
        solver_info["wall_time_sec"] = float(time.perf_counter() - wall0)
        solver_info["diagnosis"] = DIAGNOSIS
        solver_info["method"] = METHOD

    return {"u": u_grid, "solver_info": solver_info, "u_initial": u0_grid}

if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.02, "beta": [6.0, 3.0], "t0": 0.0, "t_end": 0.1, "dt": 0.02, "time": True},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
