import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """
    Solve the convection-diffusion equation:
    -eps * laplacian(u) + beta . grad(u) = f  in Omega
    u = g  on dOmega

    Manufactured solution: u = cos(pi*x)*sin(pi*y)
    eps = 0.2, beta = [0.8, 0.3], Pe ~ 4.3 (low)
    """
    t0 = time.time()

    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    eps_val = float(pde.get("diffusion", 0.2)) if isinstance(
        pde.get("diffusion", 0.2), (int, float, np.floating)) else 0.2
    beta_raw = pde.get("velocity", [0.8, 0.3])
    if isinstance(beta_raw, (list, tuple, np.ndarray)):
        beta_arr = np.array([float(b) for b in beta_raw])
    else:
        beta_arr = np.array([0.8, 0.3])

    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = int(grid_spec["nx"])
    ny_out = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    # Solver parameters - P3 with moderate mesh for high accuracy
    mesh_res = 120
    element_degree = 3
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution: u = cos(pi*x)*sin(pi*y)
    u_exact = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])

    # Compute source term analytically
    # laplacian(u) = -2*pi^2*cos(pi*x)*sin(pi*y)
    laplacian_u = -2.0 * pi**2 * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    # grad(u):
    # du/dx = -pi*sin(pi*x)*sin(pi*y)
    # du/dy = pi*cos(pi*x)*cos(pi*y)
    grad_u_x = -pi * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_y = pi * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    # beta . grad(u)
    beta_dot_grad_u = beta_arr[0] * grad_u_x + beta_arr[1] * grad_u_y
    # f = -eps*laplacian(u) + beta . grad(u)
    f_source = -eps_val * laplacian_u + beta_dot_grad_u

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    beta_ufl = ufl.as_vector(beta_arr)

    # Standard Galerkin (sufficient for low Pe ~ 4.3)
    a = (eps_val * ufl.inner(ufl.grad(u), ufl.grad(v))
         + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v)) * ufl.dx
    L = ufl.inner(f_source, v) * ufl.dx

    # Boundary conditions: u = cos(pi*x)*sin(pi*y) on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct LU
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1  # Direct solver

    # Evaluate on output grid
    xmin, xmax, ymin, ymax = float(bbox[0]), float(
        bbox[1]), float(bbox[2]), float(bbox[3])
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    error_form = fem.form(
        ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    # H1 semi-norm error
    grad_error = ufl.grad(u_sol) - ufl.grad(u_exact)
    h1_error_form = fem.form(ufl.inner(grad_error, grad_error) * ufl.dx)
    h1_error_local = fem.assemble_scalar(h1_error_form)
    h1_error = np.sqrt(
        domain.comm.allreduce(h1_error_local, op=MPI.SUM))

    wall_time = time.time() - t0

    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"H1 error: {h1_error:.6e}")
        print(f"Wall time: {wall_time:.3f}s")
        print(
            f"u_grid shape: {u_grid.shape}, "
            f"min: {np.nanmin(u_grid):.6e}, max: {np.nanmax(u_grid):.6e}")

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "diffusion": 0.2,
            "velocity": [0.8, 0.3],
        },
        "output": {
            "grid": {
                "nx": 65,
                "ny": 65,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
