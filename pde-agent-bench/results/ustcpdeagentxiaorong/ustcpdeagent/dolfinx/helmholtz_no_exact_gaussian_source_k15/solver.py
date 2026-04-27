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
    pde = case_spec.get("pde", {})
    k_val = float(pde.get("wavenumber", 15.0))

    # Output grid specification
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]

    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    # Mesh parameters - use high resolution for accuracy
    elem_degree = 3
    N = 250

    comm = MPI.COMM_WORLD

    # Create mesh
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Source term: f = 10*exp(-80*((x-0.35)**2 + (y-0.55)**2))
    f_expr = 10.0 * ufl.exp(
        -80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2)
    )

    # Wavenumber squared
    k2 = fem.Constant(domain, ScalarType(k_val ** 2))

    # Bilinear form: a(u,v) = integral (grad(u).grad(v) - k^2 u v) dx
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k2 * ufl.inner(u, v)) * ufl.dx

    # Linear form: L(v) = integral f v dx
    L = f_expr * v * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve using direct LU (indefinite system - iterative solvers can struggle)
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

    iterations = 1

    # Sample solution onto the output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_3d = np.zeros((nx_out * ny_out, 3))
    pts_3d[:, 0] = XX.ravel()
    pts_3d[:, 1] = YY.ravel()

    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, pts_3d
    )

    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full(pts_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "type": "helmholtz",
            "wavenumber": 15.0,
        },
        "domain": {
            "type": "rectangle",
            "bounds": [0.0, 1.0, 0.0, 1.0],
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()

    print(f"Wall time: {t1 - t0:.3f} s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
    assert result["u"].shape == (100, 100), f"Wrong shape: {result['u'].shape}"
    assert not np.all(result["u"] == 0), "Solution is all zeros!"
    print("All checks passed.")
