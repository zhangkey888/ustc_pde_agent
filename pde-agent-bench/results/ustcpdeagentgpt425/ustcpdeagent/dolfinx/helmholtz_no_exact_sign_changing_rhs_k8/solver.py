import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _rhs_expr(x):
    return np.cos(4.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _sample_on_grid(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values_local[np.array(eval_map, dtype=np.int32)] = np.real(vals).reshape(-1)

    values_send = np.where(np.isnan(values_local), -np.inf, values_local)
    values_global = np.empty_like(values_send)
    msh.comm.Allreduce(values_send, values_global, op=MPI.MAX)
    values_global[np.isneginf(values_global)] = 0.0

    return values_global.reshape(ny, nx)


def _assemble_and_solve(mesh_resolution=96, element_degree=2, k_value=8.0, rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    f = ufl.cos(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    k2 = ScalarType(k_value * k_value)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k2 * u * v) * ufl.dx
    L = f * v * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    used_ksp = "gmres"
    used_pc = "ilu"

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve failed")
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        used_ksp = "preonly"
        used_pc = "lu"

    uh.x.scatter_forward()

    # Accuracy verification module: compute algebraic residual norm ||A u - b||
    Au = b.duplicate()
    A.mult(uh.x.petsc_vec, Au)
    Au.axpy(-1.0, b)
    residual_l2 = Au.norm()

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "residual_l2": float(residual_l2),
    }

    return uh, solver_info


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    output = case_spec["output"]["grid"]
    nx = int(output["nx"])
    ny = int(output["ny"])
    bbox = output["bbox"]

    k_value = float(
        pde.get("wavenumber", case_spec.get("wavenumber", case_spec.get("k", 8.0)))
    )

    # Chosen to balance accuracy and speed robustly for the provided time budget.
    mesh_resolution = 96
    element_degree = 2
    rtol = 1e-10

    uh, solver_info = _assemble_and_solve(
        mesh_resolution=mesh_resolution,
        element_degree=element_degree,
        k_value=k_value,
        rtol=rtol,
    )

    u_grid = _sample_on_grid(uh, bbox=bbox, nx=nx, ny=ny)

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
