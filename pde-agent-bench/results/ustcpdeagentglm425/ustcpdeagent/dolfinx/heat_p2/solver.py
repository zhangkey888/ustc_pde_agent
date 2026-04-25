import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa_val = float(pde["coefficients"]["kappa"])
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])
    dt_suggested = float(pde["time"]["dt"])
    time_scheme = pde["time"]["scheme"]

    output_spec = case_spec["output"]
    grid = output_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    mesh_res = 80
    elem_degree = 2
    dt = 0.003

    n_steps = int(round((t_end - t0) / dt))
    actual_dt = (t_end - t0) / n_steps

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    t_const = fem.Constant(domain, ScalarType(t0))
    kappa_const = fem.Constant(domain, ScalarType(kappa_val))

    x = ufl.SpatialCoordinate(domain)

    f_expr = -ufl.exp(-t_const) * (x[0]**2 + x[1]**2 + 4.0 * kappa_val)
    g_expr = ufl.exp(-t_const) * (x[0]**2 + x[1]**2)

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u = fem.Function(V)
    u_n = fem.Function(V)

    t_const.value = ScalarType(t0)
    u_n.interpolate(fem.Expression(g_expr, V.element.interpolation_points))

    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    a_form = ufl.inner(u_trial, v) * ufl.dx + actual_dt * kappa_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L_form = ufl.inner(u_n, v) * ufl.dx + actual_dt * ufl.inner(f_func, v) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol)

    b = petsc.create_vector(L_compiled.function_spaces)

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    def sample_function(u_func, points):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)

        u_values = np.full((points.shape[1],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values

    u_initial_grid = sample_function(u_n, pts).reshape(ny_out, nx_out)

    total_iterations = 0
    t = t0

    for step in range(n_steps):
        t = t0 + (step + 1) * actual_dt

        t_const.value = ScalarType(t)

        g_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
        f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        petsc.set_bc(b, [bc])

        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = u.x.array[:]
        u_n.x.scatter_forward()

    u_grid = sample_function(u, pts).reshape(ny_out, nx_out)

    t_const.value = ScalarType(t_end)
    u_exact_final = ufl.exp(-t_const) * (x[0]**2 + x[1]**2)
    error_expr = ufl.inner(u - u_exact_final, u - u_exact_final) * ufl.dx
    error_form = fem.form(error_expr)
    error_local = fem.assemble_scalar(error_form)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    if domain.comm.rank == 0:
        print(f"L2 error at t={t_end}: {error_L2:.6e}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
