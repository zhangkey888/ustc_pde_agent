import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: helmholtz
# ```

def _sample_function_on_grid(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, -1.0e300, dtype=np.float64)
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
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        vals = np.real(np.asarray(vals).reshape(-1))
        local_vals[np.asarray(ids, dtype=np.int32)] = vals

    global_vals = np.empty_like(local_vals)
    msh.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
    global_vals[global_vals < -1.0e200] = 0.0
    return global_vals.reshape((ny, nx))

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    k = float(pde.get("k", 10.0))

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 96))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    ksp_type = str(case_spec.get("solver", {}).get("ksp_type", "gmres"))
    pc_type = str(case_spec.get("solver", {}).get("pc_type", "ilu"))
    rtol = float(case_spec.get("solver", {}).get("rtol", 1e-10))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact_expr = ufl.sin(3.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    f_expr = (((3.0 * pi) ** 2 + (2.0 * pi) ** 2) - k ** 2) * u_exact_expr

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

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
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=10000)

    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve failed")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()

    err_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ref_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))
    ref_L2 = np.sqrt(comm.allreduce(ref_sq_local, op=MPI.SUM))
    rel_L2 = err_L2 / ref_L2 if ref_L2 > 0 else err_L2

    u_grid = _sample_function_on_grid(uh, msh, nx_out, ny_out, bbox)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "relative_L2_error": float(rel_L2),
            "absolute_L2_error": float(err_L2),
        },
    }
