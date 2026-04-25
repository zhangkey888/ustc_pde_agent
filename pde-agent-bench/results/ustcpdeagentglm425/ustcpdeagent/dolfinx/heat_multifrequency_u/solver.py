import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    t0 = pde["time"]["t0"]
    t_end = pde["time"]["t_end"]
    dt_suggested = pde["time"].get("dt", 0.01)
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    mesh_res = 40
    elem_degree = 2
    dt = dt_suggested
    n_steps = int(round((t_end - t0) / dt))
    kappa = 1.0
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_uf = ufl.exp(-t_const) * (ufl.sin(pi*x[0])*ufl.sin(pi*x[1]) + 0.2*ufl.sin(6*pi*x[0])*ufl.sin(6*pi*x[1]))
    f_uf = ufl.exp(-t_const) * ((2*pi**2 - 1)*ufl.sin(pi*x[0])*ufl.sin(pi*x[1]) + 0.2*(72*pi**2 - 1)*ufl.sin(6*pi*x[0])*ufl.sin(6*pi*x[1]))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    t_const.value = PETSc.ScalarType(t0)
    u_n.interpolate(fem.Expression(u_exact_uf, V.element.interpolation_points))
    a_form = ufl.inner(u_trial, v)*ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v))*ufl.dx
    L_form = ufl.inner(u_n, v)*ufl.dx + dt * ufl.inner(f_uf, v)*ufl.dx
    a = fem.form(a_form)
    L = fem.form(L_form)
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)
    ksp.setFromOptions()
    u_sol = fem.Function(V)
    b = petsc.create_vector(L.function_spaces)
    total_iterations = 0
    t = t0
    for step in range(n_steps):
        t += dt
        t_const.value = PETSc.ScalarType(t)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L)
        petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    if comm.size > 1:
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=MPI.SUM)
        u_values = np.where(np.isnan(u_global), 0.0, u_global)
    else:
        u_values = np.where(np.isnan(u_values), 0.0, u_values)
    u_grid = u_values.reshape(ny_out, nx_out)
    t_const.value = PETSc.ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_uf, V.element.interpolation_points))
    diff = u_sol - u_exact_func
    L2_error_sq = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    L2_error = np.sqrt(domain.comm.allreduce(L2_error_sq, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {L2_error:.6e}")
    t_const.value = PETSc.ScalarType(t0)
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u_exact_uf, V.element.interpolation_points))
    u_init_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals_i = u_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_i.flatten()
    if comm.size > 1:
        u_init_global = np.zeros_like(u_init_values)
        comm.Allreduce(u_init_values, u_init_global, op=MPI.SUM)
        u_init_values = np.where(np.isnan(u_init_global), 0.0, u_init_global)
    else:
        u_init_values = np.where(np.isnan(u_init_values), 0.0, u_init_values)
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
