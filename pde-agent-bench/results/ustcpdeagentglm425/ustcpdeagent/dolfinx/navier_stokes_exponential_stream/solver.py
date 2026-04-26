import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def solve(case_spec: dict) -> dict:
    """
    Solve steady incompressible Navier-Stokes with manufactured solution.
    Strong form: u·∇u - ν∇²u + ∇p = f,  ∇·u = 0
    """
    comm = MPI.COMM_WORLD

    nu = float(case_spec["pde"]["params"]["nu"])

    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 256

    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                     cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi

    # Exact velocity
    u1_ex = pi_val * ufl.exp(2 * x[0]) * ufl.cos(pi_val * x[1])
    u2_ex = -2 * ufl.exp(2 * x[0]) * ufl.sin(pi_val * x[1])
    u_ex = ufl.as_vector([u1_ex, u2_ex])

    # Exact pressure
    p_ex = ufl.exp(x[0]) * ufl.cos(pi_val * x[1])

    # Source term: f = (u·∇)u - ν∇²u + ∇p
    conv = ufl.grad(u_ex) * u_ex
    diff = -nu * ufl.div(ufl.grad(u_ex))
    grad_p = ufl.grad(p_ex)
    f_source = conv + diff + grad_p

    # Current solution and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Weak form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f_source, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Boundary conditions: velocity on entire boundary from exact solution
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(ufl.as_vector([u1_ex, u2_ex]), V.element.interpolation_points)
    )

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Pressure pinning at corner (0,0) to remove nullspace
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Step 1: Solve Stokes as initial guess
    (u_t, p_t) = ufl.TrialFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_t), ufl.grad(v)) * ufl.dx
        - p_t * ufl.div(v) * ufl.dx
        + ufl.div(u_t) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_source, v) * ufl.dx

    stokes_prob = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_prob.solve()
    w_stokes.x.scatter_forward()

    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # Step 2: Solve nonlinear Navier-Stokes with Newton
    ns_prob = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )

    w_sol = ns_prob.solve()
    w.x.scatter_forward()

    # Get iteration info
    try:
        snes = ns_prob._snes
        snes_it = snes.getIterationNumber()
        ksp_it = snes.getLinearSolveIterations()
    except Exception:
        snes_it = 0
        ksp_it = 0

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Sample velocity magnitude on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    pts[2] = 0.0

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    # Compute velocity magnitude
    u_mag = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    # Handle parallel: replace nan with 0 and reduce
    u_mag_clean = np.nan_to_num(u_mag, nan=0.0)
    u_mag_global = comm.allreduce(u_mag_clean, op=MPI.SUM)

    # L2 error verification
    u_diff = ufl.as_vector([u1_ex - u[0], u2_ex - u[1]])
    p_diff = p_ex - p

    L2_u_err_sq = fem.assemble_scalar(fem.form(ufl.inner(u_diff, u_diff) * ufl.dx))
    L2_u_ex_sq = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    L2_p_err_sq = fem.assemble_scalar(fem.form(p_diff ** 2 * ufl.dx))
    L2_p_ex_sq = fem.assemble_scalar(fem.form(p_ex ** 2 * ufl.dx))

    L2_u_err_sq = comm.allreduce(L2_u_err_sq, op=MPI.SUM)
    L2_u_ex_sq = comm.allreduce(L2_u_ex_sq, op=MPI.SUM)
    L2_p_err_sq = comm.allreduce(L2_p_err_sq, op=MPI.SUM)
    L2_p_ex_sq = comm.allreduce(L2_p_ex_sq, op=MPI.SUM)

    rel_u = np.sqrt(L2_u_err_sq / L2_u_ex_sq) if L2_u_ex_sq > 0 else np.sqrt(L2_u_err_sq)
    rel_p = np.sqrt(L2_p_err_sq / L2_p_ex_sq) if L2_p_ex_sq > 0 else np.sqrt(L2_p_err_sq)

    if comm.rank == 0:
        print(f"L2 velocity error: {rel_u:.6e}")
        print(f"L2 pressure error: {rel_p:.6e}")
        print(f"Newton iterations: {snes_it}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": ksp_it,
        "nonlinear_iterations": [snes_it],
    }

    return {
        "u": u_mag_global if comm.size > 1 else u_mag,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"params": {"nu": 0.15}, "time": None},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}}
    }
    import time as timer
    t0 = timer.time()
    result = solve(case_spec)
    t1 = timer.time()
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {np.nanmin(result['u']):.6e}, {np.nanmax(result['u']):.6e}")
    print(f"solver_info: {result['solver_info']}")
