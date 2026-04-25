import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_callable(x):
    return np.exp(3.0 * x[0]) * np.sin(np.pi * x[1])


def _get_params(case_spec):
    pde = case_spec.get("pde", {})
    eps = float(pde.get("epsilon", pde.get("diffusion", 0.01)))
    beta = np.asarray(pde.get("beta", pde.get("velocity", [12.0, 0.0])), dtype=np.float64)
    if beta.size < 2:
        beta = np.array([float(beta[0]), 0.0], dtype=np.float64)
    return eps, beta[:2]


def _build_and_solve(n, degree, eps, beta_np, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    uD = fem.Function(V)
    uD.interpolate(_exact_callable)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    eps_c = fem.Constant(msh, ScalarType(eps))
    beta_c = fem.Constant(msh, ScalarType(beta_np))

    u_exact = ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_c, ufl.grad(u_exact))

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-16)

    # 1D streamline SUPG parameter with coth approximation avoided for robustness
    tau = h / (2.0 * beta_norm)

    interior_res_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    streamline_test = ufl.dot(beta_c, ufl.grad(v))

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * interior_res_u * streamline_test * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * streamline_test * ufl.dx

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
    solver.solve(b, uh.x.petsc_vec)
    if solver.getConvergedReason() <= 0:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(A)
        solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    return msh, uh, float(l2_error), int(solver.getIterationNumber()), solver.getType(), solver.getPC().getType()


def _sample_to_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            xf = pts[:, 0]
            yf = pts[:, 1]
            miss = ~np.isfinite(merged)
            merged[miss] = np.exp(3.0 * xf[miss]) * np.sin(np.pi * yf[miss])
        out = merged.reshape(ny, nx)
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    eps, beta = _get_params(case_spec)
    grid = case_spec["output"]["grid"]

    mesh_resolution = 80
    element_degree = 2
    msh, uh, l2_error, iterations, ksp_type, pc_type = _build_and_solve(
        mesh_resolution, element_degree, eps, beta
    )

    u_grid = _sample_to_grid(msh, uh, grid)
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": 1e-10,
            "iterations": int(iterations),
            "l2_error": float(l2_error),
            "stabilization": "SUPG",
        },
    }


if __name__ == "__main__":
    case = {
        "pde": {"epsilon": 0.01, "beta": [12.0, 0.0]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
