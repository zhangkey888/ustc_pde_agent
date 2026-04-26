import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(values, root=0)
    if comm.rank == 0:
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(out) & (~np.isnan(arr))
            out[mask] = arr[mask]
        return np.nan_to_num(out, nan=0.0)
    return values


def _sample_to_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    if u_func.function_space.mesh.comm.rank == 0:
        return vals.reshape(ny, nx)
    return None


def _solve_once(n, degree, epsilon, beta, tau_scale, rtol, ksp_type, pc_type):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    eps_c = fem.Constant(msh, ScalarType(epsilon))
    beta_c = fem.Constant(msh, np.array(beta, dtype=ScalarType))
    f_expr = ufl.exp(ScalarType(-200.0) * ((x[0] - ScalarType(0.3)) ** 2 + (x[1] - ScalarType(0.7)) ** 2))

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    beta_norm = ufl.sqrt(ufl.inner(beta_c, beta_c) + ScalarType(1e-16))
    h = ufl.CellDiameter(msh)
    tau = ScalarType(tau_scale) * h / (2.0 * beta_norm)
    a += tau * ufl.inner(beta_c, ufl.grad(v)) * (-eps_c * ufl.div(ufl.grad(u)) + ufl.inner(beta_c, ufl.grad(u))) * ufl.dx
    L += tau * ufl.inner(beta_c, ufl.grad(v)) * f_expr * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    return uh, {
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.01))
    beta = case_spec.get("pde", {}).get("beta", [10.0, 5.0])
    grid = case_spec["output"]["grid"]

    start = time.time()
    time_budget = 50.835

    candidates = [96, 128, 160, 192]
    best_grid = None
    best_meta = None
    prev_grid = None
    conv_change = None

    for n in candidates:
        if time.time() - start > 0.9 * time_budget:
            break
        uh, meta = _solve_once(n=n, degree=1, epsilon=epsilon, beta=beta,
                               tau_scale=1.0, rtol=1e-10, ksp_type="gmres", pc_type="ilu")
        sampled = _sample_to_grid(uh, grid)
        if rank == 0:
            if prev_grid is not None:
                conv_change = float(np.linalg.norm(sampled - prev_grid) / max(np.linalg.norm(sampled), 1e-14))
            prev_grid = sampled
            best_grid = sampled
            best_meta = meta

    if rank == 0:
        solver_info = dict(best_meta)
        solver_info["verification"] = {
            "grid_convergence_relative_change": conv_change,
            "supg": True,
        }
        return {"u": np.asarray(best_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": {}}


if __name__ == "__main__":
    case = {
        "pde": {"epsilon": 0.01, "beta": [10.0, 5.0]},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
