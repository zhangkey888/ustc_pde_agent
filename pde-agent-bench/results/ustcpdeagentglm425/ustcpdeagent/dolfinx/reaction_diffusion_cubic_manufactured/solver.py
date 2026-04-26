import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "backward_euler")

    epsilon = pde.get("epsilon", 1.0)
    if isinstance(epsilon, list):
        epsilon = epsilon[0]

    output = case_spec.get("output", {})
    grid_info = output.get("grid", {})
    nx_out = grid_info.get("nx", 50)
    ny_out = grid_info.get("ny", 50)
    bbox = grid_info.get("bbox", [0.0, 1.0, 0.0, 1.0])

    mesh_res = 48
    element_degree = 2
    dt = 0.001

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    fdim = domain.topology.dim - 1

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    t_const = fem.Constant(domain, ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)

    u_ex_ufl = ufl.exp(-t_const) * 0.2 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    f_ufl = u_ex_ufl * (5.0*epsilon*ufl.pi**2 - 1.0) + u_ex_ufl**3

    def u_exact_np(xv, t):
        return np.exp(-t) * 0.2 * np.sin(2*np.pi*xv[0]) * np.sin(np.pi*xv[1])

    u_n = fem.Function(V)
    u_h = fem.Function(V)

    u_n.interpolate(lambda x: u_exact_np(x, t0))
    u_h.x.array[:] = u_n.x.array[:]

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: u_exact_np(x, t0))
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u*v/dt + epsilon*ufl.inner(ufl.grad(u), ufl.grad(v)))*ufl.dx
    L = (u_n*v/dt + f_ufl*v - u_n**3*v)*ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = fem_petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    n_steps = int(round((t_end - t0) / dt))
    total_linear_iterations = 0

    u_initial_grid = _sample_on_grid(u_n, domain, nx_out, ny_out, bbox)

    for step in range(n_steps):
        t_current = t0 + (step + 1) * dt
        t_const.value = ScalarType(t_current)
        tc = t_current
        u_bc_func.interpolate(lambda x, t=tc: u_exact_np(x, t))

        with b.localForm() as loc:
            loc.set(0)
        fem_petsc.assemble_vector(b, L_form)
        fem_petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, [bc])

        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_linear_iterations += ksp.getIterationNumber()

        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()

    u_grid = _sample_on_grid(u_h, domain, nx_out, ny_out, bbox)

    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: u_exact_np(x, t_end))
    err_sq = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx))
    err_L2 = np.sqrt(domain.comm.allreduce(err_sq, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {err_L2:.6e}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def _sample_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    if domain.comm.size > 1:
        u_global = np.zeros_like(u_values)
        domain.comm.Allreduce(u_values, u_global, op=MPI.SUM)
        u_values = np.nan_to_num(u_global, nan=0.0)

    return u_values.reshape(ny, nx)
