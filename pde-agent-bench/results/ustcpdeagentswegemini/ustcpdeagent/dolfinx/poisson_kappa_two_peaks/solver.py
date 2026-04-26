import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

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
# special_notes: manufactured_solution, variable_coeff
# ```

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


def _sample_function_on_grid(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    global_vals = np.empty_like(local_vals)
    comm.Allreduce(local_vals, global_vals, op=MPI.MAX)

    if np.isnan(global_vals).any():
        nan_mask = np.isnan(global_vals)
        global_vals[nan_mask] = 0.0

    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    # Accuracy-first parameter choice within strict runtime budget
    mesh_resolution = int(case_spec.get("solver_options", {}).get("mesh_resolution", 72))
    element_degree = int(case_spec.get("solver_options", {}).get("element_degree", 2))
    ksp_type = str(case_spec.get("solver_options", {}).get("ksp_type", "cg"))
    pc_type = str(case_spec.get("solver_options", {}).get("pc_type", "hypre"))
    rtol = float(case_spec.get("solver_options", {}).get("rtol", 1.0e-10))

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact_ufl = ufl.exp(ScalarType(0.5) * x[0]) * ufl.sin(ScalarType(2.0) * pi * x[1])
    kappa = (
        ScalarType(1.0)
        + ScalarType(15.0) * ufl.exp(
            ScalarType(-200.0) * ((x[0] - ScalarType(0.25)) ** 2 + (x[1] - ScalarType(0.25)) ** 2)
        )
        + ScalarType(15.0) * ufl.exp(
            ScalarType(-200.0) * ((x[0] - ScalarType(0.75)) ** 2 + (x[1] - ScalarType(0.75)) ** 2)
        )
    )

    f = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_case_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 2000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification
    err_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    ref_form = fem.form((u_exact_ufl) ** 2 * ufl.dx)
    l2_err_local = fem.assemble_scalar(err_form)
    l2_ref_local = fem.assemble_scalar(ref_form)
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / l2_ref if l2_ref > 0 else l2_err

    u_grid = _sample_function_on_grid(uh, msh, case_spec["output"]["grid"])

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "relative_l2_error": rel_l2_err,
        "wall_time_sec": time.perf_counter() - t0,
    }

    return {"u": u_grid, "solver_info": solver_info}
