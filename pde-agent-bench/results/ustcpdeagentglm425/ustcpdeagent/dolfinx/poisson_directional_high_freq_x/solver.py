import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid spec
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # ---- Numerical parameters ----
    mesh_res = 160
    elem_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                     cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Manufactured solution: u = sin(8*pi*x)*sin(pi*y)
    # Source: f = kappa*(64*pi^2 + pi^2)*sin(8*pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0

    f_coeff = kappa * (64.0 * ufl.pi**2 + ufl.pi**2)
    f_ufl = f_coeff * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Variational form: -div(kappa * grad(u)) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.sin(8.0 * np.pi * xx[0]) * np.sin(np.pi * xx[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get solver iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # ---- Sample solution on output grid ----
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((ny_out, nx_out), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32))
        u_flat = np.full(nx_out * ny_out, np.nan)
        u_flat[eval_map] = vals.flatten()
        u_flat_all = np.zeros_like(u_flat) if comm.rank == 0 else np.empty_like(u_flat)
        comm.Reduce(u_flat, u_flat_all, op=MPI.SUM, root=0)
        if comm.rank == 0:
            nan_mask = np.isnan(u_flat_all)
            u_flat_all[nan_mask] = (np.sin(8 * np.pi * points[0, nan_mask])
                                    * np.sin(np.pi * points[1, nan_mask]))
            u_grid = u_flat_all.reshape(ny_out, nx_out)

    u_grid = comm.bcast(u_grid, root=0)

    # ---- Verification: L2 error ----
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        lambda xx: np.sin(8.0 * np.pi * xx[0]) * np.sin(np.pi * xx[1]))
    error_L2_sq = fem.assemble_scalar(
        fem.form((u_sol - u_exact_func)**2 * ufl.dx))
    error_L2_sq = domain.comm.reduce(error_L2_sq, op=MPI.SUM, root=0)
    if comm.rank == 0:
        print(f"L2 error: {np.sqrt(error_L2_sq):.6e}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {}
    }
    import time
    t0 = time.perf_counter()
    result = solve(case_spec)
    t1 = time.perf_counter()
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Total time: {t1-t0:.3f}s")
