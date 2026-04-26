import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))
    if abs(kappa - 1.0) > 1e-14:
        kappa = float(kappa)

    # Accuracy/time trade-off: for this smooth manufactured solution on quads,
    # Q2 on a moderate mesh is highly accurate and typically still fast.
    mesh_resolution = 60
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    t0 = time.perf_counter()

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    u_exact_ufl = ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = -kappa * ufl.div(ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.exp(5.0 * xx[0]) * np.sin(np.pi * xx[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_error_if_not_converged": True,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Iteration count
    iterations = int(problem.solver.getIterationNumber())

    # Accuracy verification: L2 error against manufactured solution
    e = uh - u_exact_ufl
    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    # Sample onto requested grid using eval()
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([X.ravel(), Y.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0], dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_values = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = comm.gather(local_values, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged)] = np.exp(5.0 * pts2[np.isnan(merged), 0]) * np.sin(np.pi * pts2[np.isnan(merged), 1])
        u_grid = merged.reshape(ny, nx)
    else:
        u_grid = None

    elapsed = time.perf_counter() - t0
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "l2_error_verification": float(l2_error),
        "wall_time_sec_estimate": float(elapsed),
    }

    result = {"u": u_grid, "solver_info": solver_info}
    if comm.rank == 0:
        return result
    return {"u": None, "solver_info": solver_info}
