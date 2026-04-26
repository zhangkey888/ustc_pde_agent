import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    k_val = float(case_spec["pde"]["params"]["k"])
    f_val = float(case_spec["pde"]["params"].get("f", 0.0))
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    mesh_res = 256
    element_degree = 3
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k_sq = fem.Constant(domain, PETSc.ScalarType(k_val**2))
    f_const = fem.Constant(domain, PETSc.ScalarType(f_val))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_sq * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_const, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    x_coord = ufl.SpatialCoordinate(domain)
    g_ufl = ufl.sin(3 * ufl.pi * x_coord[0]) + ufl.cos(2 * ufl.pi * x_coord[1])
    u_bc.interpolate(fem.Expression(g_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    t0 = time.time()
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    t1 = time.time()

    iterations = problem.solver.getIterationNumber()

    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Verification: boundary check
    expected_bottom = np.sin(3 * np.pi * xs) + np.cos(2 * np.pi * ymin)
    bc_error = np.nanmax(np.abs(u_grid[0, :] - expected_bottom))
    print(f"Boundary check error: {bc_error:.2e}")
    print(f"Solve time: {t1-t0:.2f}s")
    print(f"u_grid shape: {u_grid.shape}, range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "helmholtz", "params": {"k": 12.0, "f": 0.0},
                "bc": {"g": "sin(3*pi*x) + cos(2*pi*y)"}, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
