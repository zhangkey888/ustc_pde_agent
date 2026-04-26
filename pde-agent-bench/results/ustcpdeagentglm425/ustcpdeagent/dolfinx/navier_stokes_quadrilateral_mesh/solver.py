import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse case_spec
    pde = case_spec["pde"]
    nu = pde["viscosity"]  # 0.1
    output = case_spec["output"]
    grid_spec = output["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution - start with moderate, will refine if time allows
    mesh_res = 32

    # Create quadrilateral mesh on unit square
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral,
    )
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Define UFL expressions for exact solution
    x = ufl.SpatialCoordinate(msh)
    pi_c = ufl.pi

    # Exact velocity
    u_ex = ufl.as_vector([
        pi_c * ufl.cos(pi_c * x[1]) * ufl.sin(pi_c * x[0]),
        -pi_c * ufl.cos(pi_c * x[0]) * ufl.sin(pi_c * x[1])
    ])

    # Exact pressure = 0 (with mesh context for ufl.grad)
    p_ex = 0.0 * x[0]

    # Compute source term f from manufactured solution
    # f = u_ex . grad(u_ex) - nu * laplacian(u_ex) + grad(p_ex)
    grad_u_ex = ufl.grad(u_ex)
    convection = ufl.dot(u_ex, ufl.grad(u_ex))
    laplacian_u_ex = ufl.div(grad_u_ex)
    grad_p_ex = ufl.grad(p_ex)
    f_expr = convection - nu * laplacian_u_ex + grad_p_ex

    # Test and trial functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Define strain and stress
    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    # Nonlinear residual: F(w) = 0
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    # Jacobian
    J = ufl.derivative(F, w)

    # ---- Boundary Conditions ----
    fdim = msh.topology.dim - 1

    # Velocity BC on entire boundary (from manufactured solution)
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)

    u_bc_func = fem.Function(V)
    u_bc_expr = ufl.as_vector([
        pi_c * ufl.cos(pi_c * x[1]) * ufl.sin(pi_c * x[0]),
        -pi_c * ufl.cos(pi_c * x[0]) * ufl.sin(pi_c * x[1])
    ])
    u_bc_func.interpolate(
        fem.Expression(u_bc_expr, V.element.interpolation_points)
    )
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Pressure pinning at corner (0,0) - needed for pure Dirichlet velocity BCs
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin),
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # ---- Step 1: Stokes initialization ----
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    a_stokes = (
        2.0 * nu * ufl.inner(eps(u_trial), eps(v)) * ufl.dx
        - ufl.inner(p_trial, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u_trial), q) * ufl.dx
    )
    L_stokes = ufl.inner(f_expr, v) * ufl.dx

    w_init = fem.Function(W)
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs, u=w_init,
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1e-10,
        },
        petsc_options_prefix="stokes_",
    )
    stokes_problem.solve()
    w_init.x.scatter_forward()

    # Initialize w with Stokes solution
    w.x.array[:] = w_init.x.array[:]
    w.x.scatter_forward()

    # ---- Step 2: Newton solve for full NS ----
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Get number of Newton iterations from SNES
    snes = problem.snes
    newton_iters = [int(snes.getIterationNumber())]

    # Get KSP iterations
    ksp = snes.ksp
    linear_iters = int(ksp.getIterationNumber())

    # ---- Extract velocity and compute magnitude on output grid ----
    u_h = w.sub(0).collapse()
    u_h_func = fem.Function(V)
    u_h_func.interpolate(u_h)

    # Build output grid points
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    # Evaluate velocity at grid points
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h_func.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32),
        )
        u_values[eval_map] = vals

    # Compute velocity magnitude
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)

    # Gather on all processes (for parallel runs, take max to avoid nan)
    from mpi4py import MPI
    u_grid_global = np.copy(u_grid)
    comm.Allreduce(MPI.IN_PLACE, u_grid_global, op=MPI.MAX)

    # Replace any remaining NaN with 0 (shouldn't happen for points inside domain)
    u_grid_global = np.nan_to_num(u_grid_global, nan=0.0)

    # ---- Compute L2 error for verification ----
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_bc_expr, V.element.interpolation_points)
    )
    error_L2 = fem.assemble_scalar(
        fem.form(ufl.inner(u_h_func - u_exact_func, u_h_func - u_exact_func) * ufl.dx)
    )
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2, op=MPI.SUM))

    u_norm = fem.assemble_scalar(
        fem.form(ufl.inner(u_exact_func, u_exact_func) * ufl.dx)
    )
    u_norm = np.sqrt(MPI.COMM_WORLD.allreduce(u_norm, op=MPI.SUM))
    rel_error = error_L2 / u_norm if u_norm > 0 else error_L2

    print(f"Mesh res: {mesh_res}, L2 error: {error_L2:.6e}, Rel error: {rel_error:.6e}")
    print(f"Newton iters: {newton_iters}, Linear iters: {linear_iters}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": linear_iters,
        "nonlinear_iterations": newton_iters,
    }

    return {"u": u_grid_global, "solver_info": solver_info}


if __name__ == "__main__":
    # Test case
    case_spec = {
        "pde": {
            "viscosity": 0.1,
            "time": None,
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.nanmax(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
