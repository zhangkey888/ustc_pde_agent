import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    nu_val = float(pde["coefficients"]["viscosity"])
    gdim = 2

    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    time_budget = case_spec.get("time_limit_s", 236.0)
    mesh_res = 80
    if time_budget > 200:
        mesh_res = 96

    degree_u = 2
    degree_p = 1
    newton_rtol = 1e-10
    newton_atol = 1e-12
    ksp_rtol_val = 1e-12

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_ex = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    f_source = -nu_val * ufl.div(ufl.grad(u_ex)) + ufl.grad(u_ex) * u_ex + ufl.grad(p_ex)

    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_source, V.element.interpolation_points))

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    def eps(u_f):
        return ufl.sym(ufl.grad(u_f))

    def sigma(u_f, p_f):
        return 2.0 * nu_val * eps(u_f) - p_f * ufl.Identity(gdim)

    F = (ufl.inner(sigma(u, p), eps(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - ufl.inner(f_func, v) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx)

    J = ufl.derivative(F, w)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Stokes initialization
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    a_stokes = (2*nu_val*ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v_test)))
                - p_trial*ufl.div(v_test) + ufl.div(u_trial)*q_test) * ufl.dx
    L_stokes = ufl.inner(f_func, v_test) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # Newton solve
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": newton_atol,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": ksp_rtol_val,
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_opts
    )

    t0 = time.time()
    w_h = problem.solve()
    w.x.scatter_forward()
    t1 = time.time()
    wall_time = t1 - t0

    # Get solver info
    snes = problem.snes
    newton_its = int(snes.getIterationNumber())
    ksp_its_total = int(snes.getLinearSolveIterations())

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # Sample on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

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

    u_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mag_flat = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        u_flat = np.zeros(nx_out * ny_out)
        u_flat[eval_map] = mag_flat
        u_grid = u_flat.reshape(ny_out, nx_out)

    # Synchronize across processes
    if comm.Get_size() > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global

    # Error computation
    u_h_func = w_h.sub(0).collapse()
    p_h_func = w_h.sub(1).collapse()
    p_ex_func = fem.Function(Q)
    p_ex_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))

    err_u_expr = ufl.inner(u_h_func - u_ex, u_h_func - u_ex) * ufl.dx
    err_p_expr = (p_h_func - p_ex_func)**2 * ufl.dx
    L2_u = np.sqrt(fem.assemble_scalar(fem.form(err_u_expr)))
    L2_p = np.sqrt(fem.assemble_scalar(fem.form(err_p_expr)))
    if comm.Get_size() > 1:
        L2_u = comm.allreduce(L2_u, op=MPI.SUM)
        L2_p = comm.allreduce(L2_p, op=MPI.SUM)

    print(f"L2 velocity error: {L2_u:.6e}")
    print(f"L2 pressure error: {L2_p:.6e}")
    print(f"Newton iterations: {newton_its}, KSP iterations: {ksp_its_total}")
    print(f"Wall time: {wall_time:.2f}s")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": ksp_rtol_val,
        "iterations": ksp_its_total,
        "nonlinear_iterations": [newton_its],
    }

    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"viscosity": 0.1}},
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "time_limit_s": 236.0
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
