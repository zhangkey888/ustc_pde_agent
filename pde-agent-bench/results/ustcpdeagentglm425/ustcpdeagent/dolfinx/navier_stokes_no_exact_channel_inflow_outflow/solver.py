import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import os
os.environ["OMP_NUM_THREADS"] = "4"

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    nu_val = float(pde["coefficients"]["viscosity"])
    f_vec = [float(v) for v in pde["source_term"]]
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    mesh_res = 256
    degree_u = 2
    degree_p = 1
    newton_rtol = 1e-8
    newton_max_it = 50
    ksp_rtol = 1e-10

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType(np.array(f_vec, dtype=np.float64)))

    # Weak form with natural do-nothing outflow BC at x=1
    F = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    bcs = []

    # Inflow x=0: u = [4*y*(1-y), 0]
    inflow_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[1])]))
    bcs.append(fem.dirichletbc(u_inflow, inflow_dofs, W.sub(0)))

    # Bottom wall y=0: u = [0, 0]
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_wall_bot = fem.Function(V)
    u_wall_bot.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_wall_bot, bottom_dofs, W.sub(0)))

    # Top wall y=1: u = [0, 0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_wall_top = fem.Function(V)
    u_wall_top.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_wall_top, top_dofs, W.sub(0)))

    # Pressure pin at corner (0,0) for numerical stability
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))

    # Step 1: Solve Stokes for initial guess
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    a_stokes = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # Step 2: Newton solve for Navier-Stokes
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-12,
        "snes_max_it": newton_max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    ns_problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )

    ns_problem.solve()
    w.x.scatter_forward()

    snes = ns_problem.solver
    newton_its = snes.getIterationNumber()
    ksp_its_total = snes.getLinearSolveIterations()

    u_h = w.sub(0).collapse()

    # Accuracy verification: divergence L2 norm
    div_u = ufl.div(u_h)
    div_L2 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(div_u, div_u) * ufl.dx)))
    print(f"[Verification] Divergence L2 norm: {div_L2:.2e}")
    print(f"[Verification] Newton iterations: {newton_its}, KSP iterations: {ksp_its_total}")
    print(f"[Verification] SNES converged reason: {snes.getConvergedReason()}")

    # Sample onto output grid
    xmin, xmax, ymin, ymax = bbox
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

    u_grid = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_eval, cells_eval)
        magnitude = np.linalg.norm(vals, axis=1)
        u_grid[eval_map] = magnitude

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": ksp_rtol,
        "iterations": ksp_its_total,
        "nonlinear_iterations": [newton_its],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"viscosity": "0.12"},
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": {
                "dirichlet": [
                    {"boundary": "x0", "value": ["4*y*(1-y)", "0.0"]},
                    {"boundary": "y0", "value": ["0.0", "0.0"]},
                    {"boundary": "y1", "value": ["0.0", "0.0"]},
                ]
            },
            "time": {"is_transient": False},
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f} s")
    print(f"Solver info: {result['solver_info']}")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
