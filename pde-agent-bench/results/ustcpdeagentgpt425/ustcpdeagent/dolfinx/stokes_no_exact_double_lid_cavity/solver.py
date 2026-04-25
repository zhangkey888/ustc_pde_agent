import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type:        stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     diffusion
# peclet_or_reynolds:   low
# solution_regularity:  boundary_layer
# bc_type:              all_dirichlet
# special_notes:        pressure_pinning
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        minres
# preconditioner:       hypre
# special_treatment:    pressure_pinning
# pde_skill:            stokes
# ```

def _const_vec_fun(V, value):
    g = fem.Function(V)
    val = np.asarray(value, dtype=PETSc.ScalarType).reshape((-1, 1))
    g.interpolate(lambda x, v=val: np.repeat(v, x.shape[1], axis=1))
    return g

def _sample_vector_function(u_func, msh, points):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    vals_local = np.zeros((points.shape[0], msh.geometry.dim), dtype=np.float64)
    hit_local = np.zeros(points.shape[0], dtype=np.int32)

    if len(pts_local) > 0:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        vals_local[np.asarray(ids_local, dtype=np.int32), :] = np.asarray(vals, dtype=np.float64)
        hit_local[np.asarray(ids_local, dtype=np.int32)] = 1

    vals_global = np.zeros_like(vals_local)
    hit_global = np.zeros_like(hit_local)
    msh.comm.Allreduce(vals_local, vals_global, op=MPI.SUM)
    msh.comm.Allreduce(hit_local, hit_global, op=MPI.SUM)

    missing = np.where(hit_global == 0)[0]
    if missing.size > 0:
        vals_global[missing, :] = 0.0
    return vals_global

def _sample_velocity_magnitude(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_vector_function(u_func, msh, pts)
    mag = np.linalg.norm(vals, axis=1).reshape((ny, nx))
    return mag

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])

    nu_val = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.3)))

    mesh_resolution = max(64, min(128, 2 * max(nx_out, ny_out)))
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    bcs = []
    for facets, val in [
        (top_facets, (1.0, 0.0)),
        (right_facets, (0.0, -0.8)),
        (left_facets, (0.0, 0.0)),
        (bottom_facets, (0.0, 0.0)),
    ]:
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        g = _const_vec_fun(V, val)
        bcs.append(fem.dirichletbc(g, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    wh = fem.Function(W)
    rtol = 1.0e-9
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("minres")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=rtol, atol=1.0e-12, max_it=5000)

    try:
        ksp.solve(b, wh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("MINRES failed")
        ksp_type = "minres"
        pc_type = "hypre"
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        rtol = 1.0e-12
        ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=1)
        ksp.solve(b, wh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()

    # Accuracy verification without exact solution:
    # compute L2 norm of discrete divergence as incompressibility residual.
    S = fem.functionspace(msh, ("Lagrange", 1))
    div_fun = fem.Function(S)
    div_fun.interpolate(fem.Expression(ufl.div(uh), S.element.interpolation_points))
    div_sq_local = fem.assemble_scalar(fem.form(div_fun * div_fun * ufl.dx))
    div_l2 = float(np.sqrt(comm.allreduce(div_sq_local, op=MPI.SUM)))

    u_grid = _sample_velocity_magnitude(uh, msh, out_grid)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "verification": {
            "divergence_l2": div_l2
        }
    }

    return {"u": u_grid, "solver_info": solver_info}
