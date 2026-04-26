import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def _compute_residual_norm(domain, V, u_sol, k_val):
    """Accuracy verification: compute relative L2 residual norm ||Au - f|| / ||f||"""
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    f_ufl = ufl.sin(6*pi*x[0]) * ufl.cos(5*pi*x[1])

    # Residual: a(u,v) - L(v)
    residual_form = (ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
                     - k_val**2 * ufl.inner(u_sol, v) * ufl.dx
                     - ufl.inner(f_ufl, v) * ufl.dx)

    r_form = fem.form(residual_form)
    r_vec = petsc.create_vector([V])
    with r_vec.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(r_vec, r_form)
    r_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # L2 norm of f for normalization
    f_norm_form = fem.form(ufl.inner(f_ufl, f_ufl) * ufl.dx)
    f_norm_sq = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(f_norm_form), op=MPI.SUM)
    f_norm = np.sqrt(max(f_norm_sq, 1e-30))

    residual_norm = r_vec.norm(PETSc.NormType.N2)
    return residual_norm / f_norm


def solve(case_spec: dict) -> dict:
    t_start = time.time()

    # Extract parameters from case_spec
    pde = case_spec["pde"]
    k_val = float(pde["parameters"]["k"])

    output = case_spec["output"]
    grid = output["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Discretization parameters
    # For k=16, wavelength ~ 2*pi/16 ~ 0.393
    # P2 elements with mesh_resolution=80 gives ~10 DOFs per wavelength
    element_degree = 2
    mesh_resolution = 80

    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Variational form: -∇²u - k²u = f
    # Weak form: ∫∇u·∇v dx - k²∫uv dx = ∫fv dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k_sq = k_val ** 2

    # Source term using UFL for exact integration
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    f_ufl = ufl.sin(6*pi*x[0]) * ufl.cos(5*pi*x[1])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_sq * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet BC: u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # Direct LU solve (most reliable for indefinite Helmholtz)
    rtol = 1e-12
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="helmholtz_"
    )

    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    t_solve = time.time()

    # Accuracy verification
    relative_residual = _compute_residual_norm(domain, V, u_sol, k_val)

    # Sample solution on the output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    # Points need to be (3, N) for dolfinx geometry
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    points[2] = 0.0

    # Point evaluation using geometry utilities
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()

    # Reshape to (ny, nx) as required
    u_grid = u_grid.reshape(ny_out, nx_out)

    t_end = time.time()

    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
    }

    # Handle time-dependent fields if present
    if "time" in pde and pde["time"].get("is_transient", False):
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "name": "helmholtz",
            "parameters": {"k": 16.0},
            "source": "sin(6*pi*x)*cos(5*pi*y)",
            "time": {"is_transient": False}
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Max |u|: {np.nanmax(np.abs(result['u'])):.10e}")
    print(f"Solver info: {result['solver_info']}")
