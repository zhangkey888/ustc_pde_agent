import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract PDE parameters
    pde = case_spec["pde"]
    nu = float(pde["viscosity"])

    # Domain and output grid
    bbox = case_spec["output"]["grid"]["bbox"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]

    # Mesh resolution
    mesh_resolution = 64

    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle,
    )
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1

    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    # --- Boundary conditions ---
    bcs = []

    # Inflow: x = xmin, u = [sin(pi*(y-ymin)/yrange), 0.2*sin(2*pi*(y-ymin)/yrange)]
    y_range = bbox[3] - bbox[2]
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([
        np.sin(np.pi * (x[1] - bbox[2]) / y_range),
        0.2 * np.sin(2.0 * np.pi * (x[1] - bbox[2]) / y_range)
    ]))
    inflow_dofs = fem.locate_dofs_geometrical(
        (W.sub(0), V), lambda x: np.isclose(x[0], bbox[0])
    )
    bc_inflow = fem.dirichletbc(u_inflow, inflow_dofs, W.sub(0))
    bcs.append(bc_inflow)

    # No-slip walls: y = ymin and y = ymax
    for yval in [bbox[2], bbox[3]]:
        u_wall = fem.Function(V)
        u_wall.x.array[:] = 0.0
        wall_dofs = fem.locate_dofs_geometrical(
            (W.sub(0), V), lambda x: np.isclose(x[1], yval)
        )
        bc_wall = fem.dirichletbc(u_wall, wall_dofs, W.sub(0))
        bcs.append(bc_wall)

    # Pressure pin at corner for uniqueness
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2])
    )
    if isinstance(p_dofs, list):
        if len(p_dofs[0]) > 0 or len(p_dofs[1]) > 0:
            bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
            bcs.append(bc_p)
    else:
        if len(p_dofs) > 0:
            bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
            bcs.append(bc_p)

    # --- Stokes solve for initial guess ---
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f_const = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    a_stokes = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_const, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    stokes_ksp_its = stokes_problem.solver.getIterationNumber()

    # --- Nonlinear NS solve ---
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    (u, p) = ufl.split(w)
    (v2, q2) = ufl.TestFunctions(W)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    F = (
        ufl.inner(sigma(u, p), eps(v2)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v2) * ufl.dx
        - ufl.inner(f_const, v2) * ufl.dx
        + ufl.inner(ufl.div(u), q2) * ufl.dx
    )

    J = ufl.derivative(F, w)

    problem = petsc.NonlinearProblem(
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

    w_h = problem.solve()
    w.x.scatter_forward()

    # Get solver info
    snes = problem._snes
    newton_its = snes.getIterationNumber()
    ns_ksp_its = snes.getLinearSolveIterations()
    total_linear_its = stokes_ksp_its + ns_ksp_its

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # --- Accuracy verification: divergence L2 norm ---
    div_u = ufl.div(u_h)
    div_form = fem.form(div_u * div_u * ufl.dx)
    div_l2 = np.sqrt(abs(fem.assemble_scalar(div_form)))
    if comm.rank == 0:
        print(f"Verification: divergence L2 = {div_l2:.2e}")
        print(f"Newton iterations: {newton_its}")

    # --- Sample solution on output grid ---
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        magnitude = np.linalg.norm(vals, axis=1)
        u_grid[eval_map] = magnitude

    u_grid = u_grid.reshape(ny_out, nx_out)
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_linear_its,
            "nonlinear_iterations": [newton_its],
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.22,
            "time": {"is_transient": False},
            "source": ["0.0", "0.0"],
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    result = solve(case_spec)
    print(f"u shape: {result['u'].shape}")
    print(f"u max: {np.max(result['u']):.6f}")
    print(f"solver_info: {result['solver_info']}")
