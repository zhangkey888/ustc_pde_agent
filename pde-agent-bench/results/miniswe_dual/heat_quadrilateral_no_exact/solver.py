import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    source_val = float(pde.get("source_term", 1.0))
    u0_val = float(pde.get("initial_condition", 0.0))
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.12))
    dt_val = float(time_params.get("dt", 0.03))
    scheme = time_params.get("scheme", "backward_euler")
    output = case_spec.get("output", {})
    nx_out = int(output.get("nx", 50))
    ny_out = int(output.get("ny", 50))

    N = 32
    element_degree = 1
    result = _solve_at_resolution(N, element_degree, kappa_val, source_val, u0_val, t_end, dt_val, scheme, nx_out, ny_out)

    u_grid = result["u_grid"]
    u_initial = np.full((nx_out, ny_out), u0_val)
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": result["total_iterations"],
        "dt": dt_val,
        "n_steps": result["n_steps"],
        "time_scheme": scheme,
    }
    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


def _solve_at_resolution(N, degree, kappa_val, source_val, u0_val, t_end, dt_val, scheme, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.full(x.shape[1], u0_val))
    u_sol = fem.Function(V)
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    f = fem.Constant(domain, ScalarType(source_val))
    dt = fem.Constant(domain, ScalarType(dt_val))
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    a = (ufl.inner(u, v) / dt + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(f, v) + ufl.inner(u_n, v) / dt) * ufl.dx
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    n_steps = int(np.ceil(t_end / dt_val))
    total_iterations = 0
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
    norm_val = np.sqrt(fem.assemble_scalar(norm_form))
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    solver.destroy()
    A.destroy()
    b.destroy()
    return {"u_grid": u_grid, "norm": norm_val, "n_steps": n_steps, "total_iterations": total_iterations}


def _evaluate_on_grid(domain, u_func, nx, ny):
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    u_values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    u_values = np.nan_to_num(u_values, nan=0.0)
    return u_values.reshape(nx, ny)
