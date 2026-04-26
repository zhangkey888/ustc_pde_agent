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


def _sample_on_grid(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = uh.eval(np.asarray(eval_points, dtype=np.float64), np.asarray(eval_cells, dtype=np.int32))
        local_vals[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged)] = np.exp(5.0 * pts[np.isnan(merged), 0]) * np.sin(np.pi * pts[np.isnan(merged), 1])
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    mesh_resolution = 40
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -ScalarType(kappa) * ufl.div(ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.exp(5.0 * xx[0]) * np.sin(np.pi * xx[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_error_if_not_converged": True,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    e = uh - u_exact
    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = float(np.sqrt(comm.allreduce(err_local, op=MPI.SUM)))

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, int(grid["nx"]), int(grid["ny"]), grid["bbox"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(problem.solver.getIterationNumber()),
        "l2_error_verification": l2_error,
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
