import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    nu = case_spec["pde"]["params"]["nu"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 256
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_ex_0 = 2*pi*ufl.cos(2*pi*x[1])*ufl.sin(pi*x[0])
    u_ex_1 = -pi*ufl.cos(pi*x[0])*ufl.sin(2*pi*x[1])
    u_ex = ufl.as_vector([u_ex_0, u_ex_1])
    p_ex = ufl.cos(pi*x[0])*ufl.sin(pi*x[1])

    def eps_f(u_e):
        return ufl.sym(ufl.grad(u_e))

    f = ufl.grad(u_ex)*u_ex - 2*nu*ufl.div(eps_f(u_ex)) + ufl.grad(p_ex)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    bcs = [bc_u]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Stokes solve for initial guess using MUMPS
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a_stokes = (
        2*nu*ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_sol = stokes_problem.solve()
    w_sol.x.scatter_forward()

    # Picard iteration with manual assembly using MUMPS
    u_k = fem.Function(V)
    u_k.interpolate(w_sol.sub(0).collapse())

    a_oseen = (
        2*nu*ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u_trial) * u_k, v) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
    )

    a_form = fem.form(a_oseen)
    L_form = fem.form(L_stokes)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-10, max_it=500)

    picard_max_it = 25
    picard_rtol = 1e-8
    picard_iterations = 0
    total_linear_iters = 0

    w_new = fem.Function(W)

    for k in range(picard_max_it):
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()

        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        ksp.solve(b, w_new.x.petsc_vec)
        w_new.x.scatter_forward()

        diff = w_new.x.petsc_vec - w_sol.x.petsc_vec
        diff_norm = diff.norm(PETSc.NormType.NORM_2)
        w_sol_norm = w_sol.x.petsc_vec.norm(PETSc.NormType.NORM_2)
        if w_sol_norm > 1e-14:
            rel_change = diff_norm / w_sol_norm
        else:
            rel_change = diff_norm

        picard_iterations += 1

        if MPI.COMM_WORLD.rank == 0:
            print(f"Picard iter {k}: rel_change={rel_change:.6e}")

        w_sol.x.array[:] = w_new.x.array[:]
        w_sol.x.scatter_forward()
        u_k.interpolate(w_sol.sub(0).collapse())

        if rel_change < picard_rtol:
            break

    newton_iterations_list = [picard_iterations]
    ksp_rtol = 1e-10

    u_collapse = w_sol.sub(0).collapse()

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_collapse.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        magnitude = np.linalg.norm(vals, axis=1)
        u_values[eval_map] = magnitude

    u_values_global = np.zeros_like(u_values)
    MPI.COMM_WORLD.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny_out, nx_out)

    # L2 error verification
    V_high = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    u_exact_func = fem.Function(V_high)
    u_exact_func.interpolate(fem.Expression(u_ex, V_high.element.interpolation_points))
    error_expr = ufl.inner(u_collapse - u_exact_func, u_collapse - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"Picard iterations: {picard_iterations}")
        print(f"Max velocity magnitude: {np.max(u_grid):.6e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": ksp_rtol,
            "iterations": total_linear_iters,
            "nonlinear_iterations": newton_iterations_list,
        }
    }
