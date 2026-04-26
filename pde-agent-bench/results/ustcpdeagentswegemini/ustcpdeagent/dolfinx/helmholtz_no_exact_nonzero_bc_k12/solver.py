import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _boundary_expr(x):
    return np.sin(3.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0].real
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_vals) & ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    # Fallback for any unresolved points (should not happen on unit square)
    unresolved = np.isnan(global_vals)
    if np.any(unresolved):
        global_vals[unresolved] = 0.0

    return global_vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
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
    # special_notes: none
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

    k = float(case_spec.get("pde", {}).get("wavenumber", 12.0))
    # Use a fairly accurate mesh while remaining safely within the time budget.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 80))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    rtol = float(case_spec.get("solver", {}).get("rtol", 1.0e-10))

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Lift nonzero Dirichlet BC: solve for w with homogeneous Dirichlet, then u = w + u_bc
    u_bc = fem.Function(V)
    u_bc.interpolate(_boundary_expr)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero_bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    g_ufl = ufl.sin(3.0 * ufl.pi * x[0]) + ufl.cos(2.0 * ufl.pi * x[1])

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = -(
        ufl.inner(ufl.grad(g_ufl), ufl.grad(v)) - (k ** 2) * g_ufl * v
    ) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[zero_bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": rtol,
        },
    )
    w_h = problem.solve()
    w_h.x.scatter_forward()

    uh = fem.Function(V)
    uh.x.array[:] = w_h.x.array + u_bc.x.array
    uh.x.scatter_forward()

    # Accuracy verification module:
    # 1) Strong residual of imposed lifted field relation in weak form
    residual_form = fem.form(
        (ufl.inner(ufl.grad(uh), ufl.grad(v)) - (k ** 2) * uh * v) * ufl.dx
    )
    # Use an H1-like seminorm of correction as a sanity metric and BC mismatch on boundary dofs
    bc_mismatch = 0.0
    if len(boundary_dofs) > 0:
        diff = uh.x.array[boundary_dofs] - u_bc.x.array[boundary_dofs]
        bc_mismatch = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    bc_mismatch = comm.allreduce(bc_mismatch, op=MPI.MAX)

    # Residual surrogate by solving assembled residual vector norm
    res_vec = petsc.create_vector(fem.form((ufl.inner(ufl.grad(uh), ufl.grad(v)) - (k ** 2) * uh * v) * ufl.dx).function_spaces)
    with res_vec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(res_vec, fem.form((ufl.inner(ufl.grad(uh), ufl.grad(v)) - (k ** 2) * uh * v) * ufl.dx))
    res_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    residual_norm = float(res_vec.norm())

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
        "verification": {
            "assembled_residual_l2": residual_norm,
            "boundary_max_mismatch": bc_mismatch,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}
