from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import numpy as np

ScalarType = PETSc.ScalarType


def _manufactured_u(x):
    return np.sin(3.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_values[np.array(eval_map, dtype=np.int32)] = vals.real

    gathered = domain.comm.gather(local_values, root=0)

    if domain.comm.rank == 0:
        final_values = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final_values[mask] = arr[mask]
        if np.isnan(final_values).any():
            raise RuntimeError("Failed to evaluate FEM solution at all requested output points.")
        return final_values.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    """
    Solve -Δu - k^2 u = f on [0,1]^2 with Dirichlet data from the manufactured solution
    u = sin(3*pi*x) sin(3*pi*y), and return solution sampled on the requested uniform grid.
    """
    comm = MPI.COMM_WORLD

    # ```DIAGNOSIS
    # equation_type: helmholtz
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: wave
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```

    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: helmholtz
    # ```

    k = 12.0

    # Accuracy-oriented choice: smooth exact solution, generous time limit, so use fine P2 mesh.
    mesh_resolution = 160
    element_degree = 2
    requested_ksp_type = "preonly"
    requested_pc_type = "lu"
    rtol = 1.0e-12

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    f_ufl = (18.0 * ufl.pi * ufl.pi - k * k) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k * k) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_u)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    petsc_options = {
        "ksp_type": requested_ksp_type,
        "pc_type": requested_pc_type,
        "ksp_rtol": rtol,
        "pc_factor_mat_solver_type": "mumps",
    }

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix="helmholtz_", petsc_options=petsc_options
        )
        uh = problem.solve()
    except Exception:
        # Fallback to default LU if MUMPS is unavailable
        petsc_options = {
            "ksp_type": requested_ksp_type,
            "pc_type": requested_pc_type,
            "ksp_rtol": rtol,
        }
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix="helmholtz_fallback_", petsc_options=petsc_options
        )
        uh = problem.solve()

    uh.x.scatter_forward()

    # Accuracy verification
    u_exact = fem.Function(V)
    u_exact.interpolate(_manufactured_u)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    linf_local = np.max(np.abs(e.x.array)) if e.x.array.size > 0 else 0.0
    linf_error = comm.allreduce(linf_local, op=MPI.MAX)

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())
    actual_ksp_type = str(ksp.getType())
    actual_pc_type = str(ksp.getPC().getType())

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    if comm.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(element_degree),
                "ksp_type": actual_ksp_type,
                "pc_type": actual_pc_type,
                "rtol": float(rtol),
                "iterations": int(iterations),
                "l2_error": float(l2_error),
                "linf_error": float(linf_error),
            },
        }
    return {}
