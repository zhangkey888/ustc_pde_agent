import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        out = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return out.reshape(ny, nx)

    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

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
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: poisson
    # ```

    # Accuracy/time balanced default.
    mesh_resolution = int(case_spec.get("solver_params", {}).get("mesh_resolution", 56))
    element_degree = int(case_spec.get("solver_params", {}).get("element_degree", 2))
    rtol = float(case_spec.get("solver_params", {}).get("rtol", 1.0e-10))

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    u_exact_ufl = ufl.exp(2.0 * x[0]) * ufl.cos(pi * x[1])
    kappa = fem.Constant(domain, ScalarType(1.0))

    # For -div(kappa grad u) = f with kappa=1
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet BC on all boundary facets using exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Robust direct solve.
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification against manufactured solution
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = np.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    # Sample solution on requested grid
    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    # Collect solver information
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
        "verification_L2_error": float(err_L2),
        "verification_rel_L2_error": float(rel_L2),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
