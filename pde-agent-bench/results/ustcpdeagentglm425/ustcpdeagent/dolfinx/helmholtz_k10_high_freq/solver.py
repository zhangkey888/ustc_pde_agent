import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k = float(case_spec["pde"]["params"]["k"])
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    mesh_resolution = 200
    element_degree = 3

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    pi = np.pi
    f_coeff = 13.0 * pi**2 - k**2
    f_expr = f_coeff * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )

    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx_out * ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))

    error_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    )
    exact_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    )
    l2_error = (np.sqrt(float(error_sq)) / np.sqrt(float(exact_sq))
                if float(exact_sq) > 0 else np.sqrt(float(error_sq)))

    if comm.rank == 0:
        print(f"Relative L2 error: {l2_error:.6e}")

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"params": {"k": 10.0}, "time": None},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max value: {np.max(result['u']):.6f}")
