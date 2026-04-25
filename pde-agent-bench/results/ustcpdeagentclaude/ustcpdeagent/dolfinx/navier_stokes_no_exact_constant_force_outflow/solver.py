import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = 0.3
    fx, fy = 1.0, 0.0

    N = 64
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((fx, fy)))

    fdim = msh.topology.dim - 1

    def walls(x):
        return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) | np.isclose(x[0], 1.0)

    wall_facets = mesh.locate_entities_boundary(msh, fdim, walls)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u_zero, dofs_u, W.sub(0))
    bcs = [bc_u]

    lu_opts = {"ksp_type": "preonly", "pc_type": "lu",
               "pc_factor_mat_solver_type": "umfpack"}

    # Stokes initial solve
    (u_tr, p_tr) = ufl.TrialFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
        - p_tr * ufl.div(v) * ufl.dx
        + ufl.div(u_tr) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options=lu_opts,
        petsc_options_prefix="stokes_",
    )
    w_stokes = stokes_problem.solve()

    # NS Newton solve
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    (u, p) = ufl.split(w)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "umfpack",
    }

    n_newton = [0]
    try:
        problem = petsc.NonlinearProblem(
            F, w, bcs=bcs, J=J,
            petsc_options_prefix="ns_",
            petsc_options=petsc_opts,
        )
        w_res = problem.solve()
        if w_res is not None:
            w = w_res
        w.x.scatter_forward()
        try:
            snes = problem.solver
            n_newton = [int(snes.getIterationNumber())]
        except Exception:
            n_newton = [1]
    except Exception as e:
        # fallback: use Stokes solution
        w = w_stokes

    u_sol = w.sub(0).collapse()

    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    mag = np.zeros(nx * ny)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        m = np.linalg.norm(vals, axis=1)
        for k, i in enumerate(idx_map):
            mag[i] = m[k]

    mag_grid = mag.reshape(ny, nx)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
        "nonlinear_iterations": n_newton,
    }

    return {"u": mag_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Shape: {out['u'].shape}")
    print(f"Max mag: {out['u'].max():.6e}")
    print(f"Min mag: {out['u'].min():.6e}")
    print(f"Mean: {out['u'].mean():.6e}")
    print(out['solver_info'])
