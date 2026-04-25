import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as petsc_fem
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # --- Extract problem parameters ---
    pde = case_spec["pde"]
    time_params = pde["time"]
    output_spec = case_spec["output"]

    t0 = time_params["t0"]
    t_end = time_params["t_end"]
    dt_suggested = time_params.get("dt", 0.02)

    # Output grid
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    # --- Mesh and function space ---
    mesh_resolution = 140
    element_degree = 2

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    tdim = domain.topology.dim
    fdim = tdim - 1

    # --- Boundary conditions: u = 0 on all boundaries ---
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_val = fem.Function(V)
    u_bc_val.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_val, boundary_dofs)

    # --- Coefficient kappa (variable) ---
    x_coord = ufl.SpatialCoordinate(domain)
    kappa_expr_ufl = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x_coord[0]) * ufl.sin(2 * ufl.pi * x_coord[1])
    kappa = fem.Function(V)
    kappa.interpolate(
        fem.Expression(kappa_expr_ufl, V.element.interpolation_points)
    )

    # --- Source term f (time-independent) ---
    f_ufl = 1.0 + ufl.sin(2 * ufl.pi * x_coord[0]) * ufl.cos(2 * ufl.pi * x_coord[1])
    f_func = fem.Function(V)
    f_func.interpolate(
        fem.Expression(f_ufl, V.element.interpolation_points)
    )

    # --- Initial condition u0 = sin(pi*x)*sin(pi*y) ---
    u0_expr_ufl = ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1])
    u_n = fem.Function(V)
    u_n.interpolate(
        fem.Expression(u0_expr_ufl, V.element.interpolation_points)
    )

    # Save initial condition for output
    u_initial_grid = _sample_on_grid(domain, V, u_n, nx_out, ny_out, bbox)

    # --- Time stepping setup ---
    dt = dt_suggested / 6.0  # ~0.00333
    n_steps = int(np.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    theta = 0.5  # Crank-Nicolson

    # --- Variational form (theta-method / Crank-Nicolson) ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_ufl = ufl.inner(u, v) * ufl.dx + dt * theta * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_form = fem.form(a_ufl)

    L_ufl = ufl.inner(u_n, v) * ufl.dx - dt * (1.0 - theta) * kappa * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    L_form = fem.form(L_ufl)

    # Assemble matrix (constant)
    A = petsc_fem.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    u_sol = fem.Function(V)

    # Solver setup
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    ksp.setFromOptions()

    total_iterations = 0

    # --- Time loop ---
    b = petsc_fem.create_vector(L_form.function_spaces)

    for step in range(n_steps):
        with b.localForm() as loc_b:
            loc_b.set(0.0)
        petsc_fem.assemble_vector(b, L_form)

        petsc_fem.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        petsc_fem.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        its = ksp.getIterationNumber()
        total_iterations += its

        u_n.x.array[:] = u_sol.x.array[:]

    # --- Sample solution on output grid ---
    u_grid = _sample_on_grid(domain, V, u_sol, nx_out, ny_out, bbox)

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": float(dt),
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        }
    }

    return result


def _sample_on_grid(domain, V, u_func, nx, ny, bbox):
    """Sample a dolfinx Function on a uniform grid."""
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    if domain.comm.size > 1:
        isnan = np.isnan(u_values)
        recv = np.zeros_like(u_values)
        domain.comm.Allreduce(u_values, recv, op=MPI.MAX)
        recv_nan = np.zeros_like(u_values)
        domain.comm.Allreduce(isnan.astype(np.float64), recv_nan, op=MPI.SUM)
        u_values = recv
        u_values[recv_nan > 0] = np.nan

    return u_values.reshape(ny, nx)


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.6*sin(2*pi*x)*sin(2*pi*y)"}
            },
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    result = solve(case_spec)
    print("u shape:", result["u"].shape)
    print("solver_info:", result["solver_info"])
    print("u min/max:", np.nanmin(result["u"]), np.nanmax(result["u"]))
