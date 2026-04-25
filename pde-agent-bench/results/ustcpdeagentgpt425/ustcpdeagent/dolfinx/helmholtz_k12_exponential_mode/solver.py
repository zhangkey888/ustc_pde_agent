from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_u_numpy(x):
    return np.exp(x[0]) * np.cos(2.0 * np.pi * x[1])


def _sample_on_grid(u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_values[np.array(eval_map, dtype=np.int32)] = vals.real

    gathered = domain.comm.gather(local_values, root=0)
    if domain.comm.rank == 0:
        result = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            result[mask] = arr[mask]
        if np.isnan(result).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return result.reshape((ny, nx))
    return None


def _compute_l2_error(u_h, u_exact_expr):
    V = u_h.function_space
    msh = V.mesh
    Verr = fem.functionspace(msh, ("Lagrange", 4))

    u_ex = fem.Function(Verr)
    u_ex.interpolate(fem.Expression(u_exact_expr, Verr.element.interpolation_points))

    u_num = fem.Function(Verr)
    u_num.interpolate(u_h)

    e = fem.Function(Verr)
    e.x.array[:] = u_num.x.array - u_ex.x.array

    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    return float(np.sqrt(msh.comm.allreduce(err_local, op=MPI.SUM)))


def _solve_helmholtz(mesh_resolution, element_degree, k_value):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.cos(2.0 * np.pi * x[1])
    f_expr = -ufl.div(ufl.grad(u_exact)) - (k_value ** 2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * u * v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _manufactured_u_numpy(X).astype(np.float64))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    preferred = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1.0e-9,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
    }
    fallback = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1.0e-12,
        "ksp_atol": 1.0e-14,
    }

    prefix = f"helmholtz_{mesh_resolution}_{element_degree}_"
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=prefix,
            petsc_options=preferred,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        its = int(problem.solver.getIterationNumber())
        reason = int(problem.solver.getConvergedReason())
        if reason <= 0:
            raise RuntimeError(f"KSP diverged with reason {reason}")
        used = preferred
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=prefix + "fallback_",
            petsc_options=fallback,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        its = int(problem.solver.getIterationNumber())
        used = fallback

    l2_error = _compute_l2_error(uh, u_exact)
    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(used["ksp_type"]),
        "pc_type": str(used["pc_type"]),
        "rtol": float(used["ksp_rtol"]),
        "iterations": its,
        "l2_error": l2_error,
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    k_value = float(case_spec.get("pde", {}).get("k", 12.0))
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    uh, info = _solve_helmholtz(56, 2, k_value)
    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": info["mesh_resolution"],
                "element_degree": info["element_degree"],
                "ksp_type": info["ksp_type"],
                "pc_type": info["pc_type"],
                "rtol": info["rtol"],
                "iterations": info["iterations"],
                "l2_error": info["l2_error"],
            },
        }
    return {"u": None, "solver_info": None}
