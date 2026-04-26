import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = float(case_spec["pde"]["coefficients"]["nu"])
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    pde_time = case_spec["pde"].get("time", {})
    is_transient = pde_time.get("is_transient", False)

    mesh_res = 96
    eps_p = 1e-6
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Boundary conditions
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.8), np.zeros(x.shape[1])]))
    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))

    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack([np.full(x.shape[1], -0.8), np.zeros(x.shape[1])]))
    bc_bottom = fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0))

    bcs = [bc_top, bc_bottom]

    # Pressure pin at corner (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs.append(bc_p)

    # Stokes solve with pressure stabilization
    a_stokes = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
        + eps_p * p_trial * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w_stokes.x.scatter_forward()

    # Initialize NS with Stokes solution
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # NS residual
    (u, p) = ufl.split(w)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        + eps_p * p * q * ufl.dx
    )

    J = ufl.derivative(F, w)

    # Newton solve
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 50,
            "snes_converged_reason": "",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    snes = problem.solver
    newton_its = int(snes.getIterationNumber())
    ksp_its = int(snes.getLinearSolveIterations())

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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

    if comm.size > 1:
        magnitude_global = np.zeros_like(magnitude)
        comm.Allreduce(magnitude, magnitude_global, op=MPI.MAX)
        magnitude = magnitude_global

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": ksp_its,
        "nonlinear_iterations": [newton_its],
    }

    if is_transient:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    return {"u": magnitude, "solver_info": solver_info}

if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.2},
            "source": {"f": ["0.0", "0.0"]},
            "time": {"is_transient": False}
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    u = result["u"]
    si = result["solver_info"]
    print(f"Shape: {u.shape}")
    print(f"Max mag: {np.nanmax(u):.6f}")
    print(f"Min mag: {np.nanmin(u):.6f}")
    print(f"Newton: {si['nonlinear_iterations']}")
    print(f"KSP its: {si['iterations']}")
    print(f"Time: {t1-t0:.2f}s")
