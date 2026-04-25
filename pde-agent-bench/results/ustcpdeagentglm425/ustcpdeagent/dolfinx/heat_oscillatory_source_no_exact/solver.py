"""Solver for transient heat equation with backward Euler time stepping."""
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
    tp = pde["time"]
    t0, t_end = float(tp["t0"]), float(tp["t_end"])

    out = case_spec["output"]
    grid = out["grid"]
    nx_out, ny_out = int(grid["nx"]), int(grid["ny"])
    bbox = [float(v) for v in grid["bbox"]]

    mesh_res = 160
    elem_deg = 2
    dt_use = 0.001
    n_steps = max(1, int(round((t_end - t0) / dt_use)))
    dt_actual = (t_end - t0) / n_steps

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                     cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x_coord = ufl.SpatialCoordinate(domain)
    f_ufl = ufl.sin(6.0 * ufl.pi * x_coord[0]) * ufl.sin(6.0 * ufl.pi * x_coord[1])

    dt_const = fem.Constant(domain, ScalarType(dt_actual))

    a_form = ufl.inner(u, v) * ufl.dx + dt_const * kappa_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    u_prev = fem.Function(V)
    u_prev.x.array[:] = 0.0

    L_form = ufl.inner(u_prev, v) * ufl.dx + dt_const * ufl.inner(f_ufl, v) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()

    u_sol = fem.Function(V)

    total_ksp_iterations = 0
    for step in range(n_steps):
        b = petsc.create_vector(L_compiled.function_spaces)
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_ksp_iterations += ksp.getIterationNumber()
        u_prev.x.array[:] = u_sol.x.array[:]

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_flat = np.zeros(nx_out * ny_out)
        for j, idx in enumerate(idx_map):
            u_flat[idx] = vals[j, 0]
        u_grid = u_flat.reshape(ny_out, nx_out)

    u_grid_global = np.zeros_like(u_grid)
    comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)

    u_initial_grid = np.zeros((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": total_ksp_iterations,
        "dt": dt_actual,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid_global,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
