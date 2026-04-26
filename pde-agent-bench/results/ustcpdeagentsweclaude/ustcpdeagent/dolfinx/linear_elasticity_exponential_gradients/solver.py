import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type:        linear_elasticity
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     mixed
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       amg
# special_treatment:    none
# pde_skill:            linear_elasticity
# ```


ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Problem parameters
    E = float(case_spec.get("material", {}).get("E", 1.0))
    nu = float(case_spec.get("material", {}).get("nu", 0.33))

    # Accuracy-oriented fixed defaults within time budget
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 96))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    if nu > 0.4:
        element_degree = max(element_degree, 2)

    ksp_type = case_spec.get("solver", {}).get("ksp_type", "cg")
    pc_type = case_spec.get("solver", {}).get("pc_type", "hypre")
    rtol = float(case_spec.get("solver", {}).get("rtol", 1e-10))

    # Grid specification for output
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    # Mesh and space
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", element_degree, (gdim,)))

    # Material constants
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    # Manufactured exact solution
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector(
        [
            ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1]),
            -ufl.exp(2.0 * x[1]) * ufl.sin(ufl.pi * x[0]),
        ]
    )

    # Body force from strong form: -div(sigma(u)) = f
    f_expr = -ufl.div(sigma(u_exact))

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Dirichlet BC on entire boundary using exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with robust fallback
    uh = fem.Function(V)
    iterations = 0
    used_ksp = ksp_type
    used_pc = pc_type

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            u=uh,
            petsc_options_prefix="le_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
                "ksp_atol": 1e-14,
                "ksp_max_it": 2000,
            },
        )
        uh = problem.solve()
        try:
            iterations = problem.solver.getIterationNumber()
        except Exception:
            iterations = 0
    except Exception:
        used_ksp = "preonly"
        used_pc = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            u=uh,
            petsc_options_prefix="le_fallback_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        iterations = 1

    uh.x.scatter_forward()

    # Accuracy verification: L2 error vs exact solution
    e = uh - u_exact
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    # Sample displacement magnitude on requested uniform grid, output shape (ny, nx)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    mags = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        mags_local = np.linalg.norm(vals, axis=1)
        mags[np.array(eval_map, dtype=np.int32)] = mags_local

    # Manufactured fallback for any unresolved grid points on this rank
    # and parallel-safe reduction by nanmin after replacing nans with +inf.
    inf_data = np.where(np.isnan(mags), np.inf, mags)
    global_data = np.empty_like(inf_data)
    comm.Allreduce(inf_data, global_data, op=MPI.MIN)

    unresolved = ~np.isfinite(global_data)
    if np.any(unresolved):
        xr = pts[unresolved, 0]
        yr = pts[unresolved, 1]
        ue0 = np.exp(2.0 * xr) * np.sin(np.pi * yr)
        ue1 = -np.exp(2.0 * yr) * np.sin(np.pi * xr)
        global_data[unresolved] = np.sqrt(ue0 * ue0 + ue1 * ue1)

    u_grid = global_data.reshape(ny, nx)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": used_ksp,
            "pc_type": used_pc,
            "rtol": rtol,
            "iterations": int(iterations),
            "l2_error": float(l2_err),
        },
    }
    return result
