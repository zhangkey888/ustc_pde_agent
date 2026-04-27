import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""

    # Extract parameters from case_spec
    k_val = case_spec["pde"]["parameters"]["k"]

    # Output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Choose mesh resolution and element degree
    # For k=15, wavelength ~ 2*pi/k ~ 0.42. With P3, we need ~10 pts/wavelength.
    element_degree = 3
    mesh_resolution = 80

    comm = MPI.COMM_WORLD

    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Manufactured solution: u = sin(2*pi*x)*sin(pi*y) + sin(pi*x)*sin(3*pi*y)
    u_exact_ufl = (
        ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
        + ufl.sin(pi * x[0]) * ufl.sin(3 * pi * x[1])
    )

    # Compute source term f from -nabla^2 u - k^2 u = f symbolically
    k_const = fem.Constant(domain, ScalarType(k_val))
    f_expr = -ufl.div(ufl.grad(u_exact_ufl)) - k_const**2 * u_exact_ufl

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Weak form: integral(grad(u).grad(v)) dx - k^2 integral(u*v) dx = integral(f*v) dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions (Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Interpolate exact solution for BC
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve using direct solver (LU) - robust for indefinite Helmholtz
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_",
    )
    u_sol = problem.solve()

    # Compute L2 error for verification
    error_form = fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {l2_error:.6e}")

    # Sample solution onto uniform output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
        },
    }

    return result


if __name__ == "__main__":
    import time

    # Construct a test case_spec
    case_spec = {
        "pde": {
            "parameters": {"k": 15.0},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    # Verify against exact solution on grid
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = (
        np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
        + np.sin(np.pi * XX) * np.sin(3 * np.pi * YY)
    )
    l2_grid_error = np.sqrt(np.mean((result["u"] - u_exact) ** 2))
    linf_grid_error = np.max(np.abs(result["u"] - u_exact))
    print(f"Grid L2 error: {l2_grid_error:.6e}")
    print(f"Grid Linf error: {linf_grid_error:.6e}")
