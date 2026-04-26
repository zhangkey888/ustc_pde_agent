import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        missing = np.isnan(merged)
        if np.any(missing):
            merged[missing] = np.sin(3.0 * np.pi * pts[missing, 0]) + np.cos(2.0 * np.pi * pts[missing, 1])
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    n = 48
    degree = 2

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    cell_name = domain.topology.cell_name()

    mixed_el = basix.ufl.mixed_element([
        basix.ufl.element("Lagrange", cell_name, degree),
        basix.ufl.element("Lagrange", cell_name, degree),
    ])
    W = fem.functionspace(domain, mixed_el)
    U, _ = W.sub(0).collapse()
    V, _ = W.sub(1).collapse()

    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(3.0 * ufl.pi * x[0]) + ufl.cos(2.0 * ufl.pi * x[1])
    v_exact = 9.0 * ufl.pi**2 * ufl.sin(3.0 * ufl.pi * x[0]) + 4.0 * ufl.pi**2 * ufl.cos(2.0 * ufl.pi * x[1])
    f_expr_ufl = 81.0 * ufl.pi**4 * ufl.sin(3.0 * ufl.pi * x[0]) + 16.0 * ufl.pi**4 * ufl.cos(2.0 * ufl.pi * x[1])

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr_ufl, V.element.interpolation_points))

    u_bc_fun = fem.Function(U)
    u_bc_fun.interpolate(lambda X: np.sin(3.0 * np.pi * X[0]) + np.cos(2.0 * np.pi * X[1]))

    v_bc_fun = fem.Function(V)
    v_bc_fun.interpolate(lambda X: 9.0 * np.pi**2 * np.sin(3.0 * np.pi * X[0]) + 4.0 * np.pi**2 * np.cos(2.0 * np.pi * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), U), fdim, boundary_facets)
    dofs_v = fem.locate_dofs_topological((W.sub(1), V), fdim, boundary_facets)
    bcs = [
        fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0)),
        fem.dirichletbc(v_bc_fun, dofs_v, W.sub(1)),
    ]

    a = (
        (ufl.inner(v, phi) - ufl.inner(ufl.grad(u), ufl.grad(phi))) * ufl.dx
        + ufl.inner(ufl.grad(v), ufl.grad(psi)) * ufl.dx
    )
    L = ufl.inner(f_fun, psi) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="biharmonic_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()

    u_exact_fun = fem.Function(U)
    u_exact_fun.interpolate(lambda X: np.sin(3.0 * np.pi * X[0]) + np.cos(2.0 * np.pi * X[1]))
    e = fem.Function(U)
    e.x.array[:] = uh.x.array - u_exact_fun.x.array
    e.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
        "l2_error": float(l2_error),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}, "pde": {"time": None}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
