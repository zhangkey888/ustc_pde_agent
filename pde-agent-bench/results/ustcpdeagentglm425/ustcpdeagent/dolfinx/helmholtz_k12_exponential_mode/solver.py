import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract PDE parameters
    pde = case_spec["pde"]
    k = float(pde.get("k", 12.0))

    # Extract output grid info
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    # Analytical source term for u = exp(x)*cos(2*pi*y)
    # laplacian(u) = (1 - 4*pi^2) * exp(x)*cos(2*pi*y)
    # f = -laplacian(u) - k^2*u = (4*pi^2 - 1 - k^2) * exp(x)*cos(2*pi*y)
    coeff_f = 4.0 * np.pi**2 - 1.0 - k**2

    # Optimal parameters: mesh=350, P3 gives L2~1.95e-11 in ~12s
    element_degree = 3
    mesh_resolution = 350

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Bilinear form: a(u, v) = integral(grad(u).grad(v) - k^2*u*v) dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx

    # Source term
    f_ufl = coeff_f * ufl.exp(x[0]) * ufl.cos(2 * ufl.pi * x[1])

    # Linear form
    L = f_ufl * v * ufl.dx

    # Dirichlet BC: u = exp(x)*cos(2*pi*y) on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    g_ufl = ufl.exp(x[0]) * ufl.cos(2 * ufl.pi * x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct solver (MUMPS for indefinite system)
    rtol = 1e-12
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(g_ufl, V.element.interpolation_points))

    L2_error_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
    )
    L2_error = np.sqrt(L2_error_sq) if comm.size == 1 else np.sqrt(comm.allreduce(L2_error_sq, op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}, mesh: {mesh_resolution}, deg: {element_degree}")

    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    pts[:, 2] = 0.0

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global

    u_grid = u_values.reshape(ny_out, nx_out)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": iterations,
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 12.0, "type": "helmholtz"},
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]},
            "field": "scalar"
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max: {np.nanmax(result['u']):.6f}, Min: {np.nanmin(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
