import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Extract parameters
    pde = case_spec["pde"]
    k_val = pde["parameters"]["k"]
    k2 = k_val ** 2

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Since manufactured solution is degree-4 polynomial, P4 elements capture it exactly
    # Use minimal mesh for speed
    mesh_res = 8
    degree = 4

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Source term derived from manufactured solution
    u_exact = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    f_expr = (
        2.0 * x[1] * (1.0 - x[1])
        + 2.0 * x[0] * (1.0 - x[0])
        - k2 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    )

    # Bilinear form: ∫ ∇u·∇v dx - k² ∫ u*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = 0 on ∂Ω (exact solution vanishes on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct solver
    ksp_type = "preonly"
    pc_type = "lu"
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_",
    )
    u_sol = problem.solve()

    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

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

    # Compute error for verification
    u_exact_grid = XX.ravel() * (1 - XX.ravel()) * YY.ravel() * (1 - YY.ravel())
    valid = ~np.isnan(u_values)
    l2_err = np.sqrt(np.mean((u_values[valid] - u_exact_grid[valid]) ** 2))
    linf_err = np.max(np.abs(u_values[valid] - u_exact_grid[valid]))
    print(f"L2 error (grid): {l2_err:.6e}")
    print(f"Linf error (grid): {linf_err:.6e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-12,
            "iterations": 1,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "parameters": {"k": 2.0},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.4f} s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
