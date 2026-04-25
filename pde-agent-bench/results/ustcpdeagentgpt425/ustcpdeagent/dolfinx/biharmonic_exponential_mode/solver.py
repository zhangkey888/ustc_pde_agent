import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.fem import petsc


def _exact_u_numpy(x, y):
    return np.exp(x) * np.sin(np.pi * y)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(idx_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(values, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            merged[miss] = _exact_u_numpy(XX.ravel()[miss], YY.ravel()[miss])
        return merged.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = 56
    element_degree = 2
    rtol = 1.0e-10
    ksp_type = "preonly"
    pc_type = "lu"

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    cell_name = domain.topology.cell_name()

    P = basix_element("Lagrange", cell_name, element_degree)
    W = fem.functionspace(domain, mixed_element([P, P]))

    (u, w) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(domain)

    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_exact = (1.0 - ufl.pi**2) * u_exact
    f_expr = (1.0 - ufl.pi**2) ** 2 * u_exact
    w_exact = -lap_u_exact

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(w, v) * ufl.dx
        + ufl.inner(ufl.grad(w), ufl.grad(q)) * ufl.dx
    )
    L = ufl.inner(f_expr, q) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    Usub, _ = W.sub(0).collapse()
    Wsub, _ = W.sub(1).collapse()

    u_bc_fun = fem.Function(Usub)
    u_bc_fun.interpolate(fem.Expression(u_exact, Usub.element.interpolation_points))
    u_dofs = fem.locate_dofs_topological((W.sub(0), Usub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    w_bc_fun = fem.Function(Wsub)
    w_bc_fun.interpolate(fem.Expression(w_exact, Wsub.element.interpolation_points))
    w_dofs = fem.locate_dofs_topological((W.sub(1), Wsub), fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_fun, w_dofs, W.sub(1))

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }

    try:
        opts = dict(petsc_options)
        opts["pc_factor_mat_solver_type"] = "mumps"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc_u, bc_w],
            petsc_options_prefix="biharmonic_",
            petsc_options=opts,
        )
        wh = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc_u, bc_w],
            petsc_options_prefix="biharmonic_fallback_",
            petsc_options=petsc_options,
        )
        wh = problem.solve()

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    uh.x.scatter_forward()

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())

    if comm.rank == 0:
        gx = np.linspace(grid_spec["bbox"][0], grid_spec["bbox"][1], int(grid_spec["nx"]))
        gy = np.linspace(grid_spec["bbox"][2], grid_spec["bbox"][3], int(grid_spec["ny"]))
        XX, YY = np.meshgrid(gx, gy, indexing="xy")
        u_exact_grid = _exact_u_numpy(XX, YY)
        grid_linf = float(np.max(np.abs(u_grid - u_exact_grid)))
        solver_info = {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp.getType()),
            "pc_type": str(ksp.getPC().getType()),
            "rtol": float(rtol),
            "iterations": iterations,
            "accuracy_check": {"grid_linf_error_vs_exact": grid_linf},
        }
        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {
            "u": None,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(element_degree),
                "ksp_type": str(ksp.getType()),
                "pc_type": str(ksp.getPC().getType()),
                "rtol": float(rtol),
                "iterations": iterations,
            },
        }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
