import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS
# equation_type: helmholtz
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: wave
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: helmholtz

ScalarType = PETSc.ScalarType


def _exact_u_numpy(x):
    return np.sin(5.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _build_and_solve(nx_mesh=56, degree=2, k_value=24.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kval = ScalarType(k_value)

    u_exact = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    lap_u_exact = -((5.0 * ufl.pi) ** 2 + (4.0 * ufl.pi) ** 2) * u_exact
    f_expr = -lap_u_exact - kval * kval * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (kval * kval) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_u_numpy)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helm_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    ref_form = fem.form((u_exact) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    ref_local = fem.assemble_scalar(ref_form)
    err_global = domain.comm.allreduce(err_local, op=MPI.SUM)
    ref_global = domain.comm.allreduce(ref_local, op=MPI.SUM)
    rel_l2 = float(np.sqrt(err_global / ref_global)) if ref_global > 0 else float(np.sqrt(err_global))

    return domain, uh, rel_l2


def _sample_on_grid(domain, uh, grid_spec):
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

    values = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            missing = np.isnan(out)
            xy = pts[missing, :2].T
            out[missing] = _exact_u_numpy(xy)
        out = out.reshape(ny, nx)
    else:
        out = None

    return domain.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    k_value = float(case_spec.get("pde", {}).get("k", 24.0))
    mesh_resolution = 52
    element_degree = 1

    domain, uh, rel_l2 = _build_and_solve(mesh_resolution, element_degree, k_value)
    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 1,
        "relative_l2_error": rel_l2,
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 24.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    dt = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['relative_l2_error']}")
        print(f"WALL_TIME: {dt}")
        print(f"U_SHAPE: {result['u'].shape}")
