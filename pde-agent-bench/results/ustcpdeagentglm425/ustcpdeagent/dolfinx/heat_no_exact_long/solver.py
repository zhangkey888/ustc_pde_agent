import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa_val = pde["coefficients"]["kappa"]

    time_info = pde["time"]
    t0 = time_info["t0"]
    t_end = time_info["t_end"]
    dt_suggested = time_info["dt"]

    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    mesh_res = 80
    element_degree = 2
    dt = dt_suggested

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    n_steps = int(round((t_end - t0) / dt))
    dt_inv = 1.0 / dt

    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(domain, ScalarType(kappa_val))

    f_func = fem.Function(V)
    f_func.interpolate(lambda x: np.cos(2*np.pi*x[0]) * np.sin(np.pi*x[1]))

    a = dt_inv * ufl.inner(u_tr, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = dt_inv * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_func, v) * ufl.dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_val = fem.Function(V)
    u_bc_val.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_val, boundary_dofs)

    u_n.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.GAMG)
    rtol = 1e-10
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    total_iterations = 0
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx_out * ny_out,), np.nan)
    p_proc, c_proc, e_map = [], [], []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            p_proc.append(points.T[i])
            c_proc.append(links[0])
            e_map.append(i)
    if len(p_proc) > 0:
        vals = u_sol.eval(np.array(p_proc), np.array(c_proc, dtype=np.int32))
        u_values[e_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    u_n_out = fem.Function(V)
    u_n_out.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
    u_init_values = np.full((nx_out * ny_out,), np.nan)
    p2, c2, em2 = [], [], []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            p2.append(points.T[i])
            c2.append(links[0])
            em2.append(i)
    if len(p2) > 0:
        vals2 = u_n_out.eval(np.array(p2), np.array(c2, dtype=np.int32))
        u_init_values[em2] = vals2.flatten()
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "gamg",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
