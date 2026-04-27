import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec - handle multiple formats
    pde = case_spec.get("pde", {})
    
    # Get viscosity
    if "coefficients" in pde and "nu" in pde["coefficients"]:
        nu_val = float(pde["coefficients"]["nu"])
    elif "viscosity" in pde:
        nu_val = float(pde["viscosity"])
    else:
        nu_val = 0.2
    
    # Get output grid
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh parameters
    N = 136
    degree_u = 2
    degree_p = 1
    eps_p = 1e-8  # pressure stabilization
    
    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Constants
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Boundary conditions
    bcs = []
    
    # Top: u = [0.8, 0.0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], ymax))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.8), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))
    
    # Bottom: u = [-0.8, 0.0]
    bot_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], ymin))
    bot_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bot_facets)
    u_bot = fem.Function(V)
    u_bot.interpolate(lambda x: np.vstack([np.full(x.shape[1], -0.8), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_bot, bot_dofs, W.sub(0)))
    
    # Pressure pinning at corner (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))
    
    # Helper function
    def eps_fn(u):
        return ufl.sym(ufl.grad(u))
    
    # --- Step 1: Stokes solve as initial guess ---
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    
    a_stokes = (
        2.0 * nu * ufl.inner(eps_fn(u_trial), eps_fn(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        + ufl.div(u_trial) * q_test * ufl.dx
        + eps_p * p_trial * q_test * ufl.dx
    )
    L_stokes = ufl.inner(f, v_test) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w_stokes.x.scatter_forward()
    
    # --- Step 2: Newton solve for full NS ---
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # NS residual (symmetric stress form with pressure stabilization)
    F_form = (
        2.0 * nu * ufl.inner(eps_fn(u), eps_fn(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        + eps_p * p * q * ufl.dx
    )
    
    J_form = ufl.derivative(F_form, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Get solver info
    snes = problem.solver
    newton_its = int(snes.getIterationNumber())
    ksp_its = int(snes.getLinearSolveIterations())
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # --- Sample on output grid ---
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, tdim)
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    # Handle NaN at boundary points
    if np.any(np.isnan(magnitude)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(magnitude.ravel())
        if np.sum(valid) > 0:
            interp = NearestNDInterpolator(
                np.column_stack([XX.ravel()[valid], YY.ravel()[valid]]),
                magnitude.ravel()[valid]
            )
            nan_mask = np.isnan(magnitude)
            nan_pts = np.column_stack([XX.ravel()[nan_mask.ravel()], YY.ravel()[nan_mask.ravel()]])
            magnitude[nan_mask] = interp(nan_pts)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": ksp_its,
        "nonlinear_iterations": [newton_its],
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.2},
            "source": {"f": ["0.0", "0.0"]},
            "time": {"is_transient": False}
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    u = result["u"]
    si = result["solver_info"]
    print(f"Wall time: {t1 - t0:.2f} s")
    print(f"Shape: {u.shape}")
    print(f"Max mag: {np.nanmax(u):.6f}")
    print(f"Min mag: {np.nanmin(u):.6f}")
    print(f"Any NaN: {np.any(np.isnan(u))}")
    print(f"Newton iters: {si['nonlinear_iterations']}")
    print(f"KSP iters: {si['iterations']}")
