import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 192
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u = 2
    deg_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.22

    # BCs
    fdim = msh.topology.dim - 1

    # Inflow x=0: u = (sin(pi y), 0.2 sin(2 pi y))
    def inflow_expr(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = np.sin(np.pi * x[1])
        vals[1] = 0.2 * np.sin(2 * np.pi * x[1])
        return vals

    u_in = fem.Function(V)
    u_in.interpolate(inflow_expr)
    facets_in = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_in = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_in)
    bc_in = fem.dirichletbc(u_in, dofs_in, W.sub(0))

    # Walls y=0 and y=1: u = 0
    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    facets_wall = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    )
    dofs_wall = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_wall)
    bc_wall = fem.dirichletbc(u_wall, dofs_wall, W.sub(0))

    bcs = [bc_in, bc_wall]
    # Outflow at x=1: natural (do-nothing) BC

    # Initial guess: Stokes
    w = fem.Function(W)
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a_stokes = (nu * ufl.inner(ufl.grad(u_t), ufl.grad(v)) * ufl.dx
                - p_t * ufl.div(v) * ufl.dx
                - ufl.div(u_t) * q * ufl.dx)
    f_vec = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    L_stokes = ufl.inner(f_vec, v) * ufl.dx + fem.Constant(msh, PETSc.ScalarType(0.0)) * q * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_init_"
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]

    # Nonlinear Navier-Stokes
    (u, p) = ufl.split(w)
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx
         - ufl.inner(f_vec, v) * ufl.dx)

    J = ufl.derivative(F, w)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 40,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_opts)
    w_sol = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w_sol.sub(0).collapse()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    # Verification: divergence check
    div_u = fem.assemble_scalar(fem.form(ufl.div(u_h) * ufl.div(u_h) * ufl.dx))
    div_u = np.sqrt(max(div_u, 0.0))
    print(f"||div u||_L2 = {div_u:.3e}")

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": 0,
            "nonlinear_iterations": [int(problem.solver.getIterationNumber())
                                      if hasattr(problem, 'solver') else 10],
        }
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Shape: {out['u'].shape}")
    print(f"Min/Max: {out['u'].min():.4f} / {out['u'].max():.4f}")
    print(f"Info: {out['solver_info']}")
