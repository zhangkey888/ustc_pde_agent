import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 192
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.18
    fdim = msh.topology.dim - 1

    # BCs
    def make_bc(marker, value):
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        ufunc = fem.Function(V)
        v = np.array(value, dtype=np.float64)
        ufunc.interpolate(lambda x: np.tile(v.reshape(2, 1), (1, x.shape[1])))
        return fem.dirichletbc(ufunc, dofs, W.sub(0))

    # Order matters: corners — apply moving lids last? Actually for cavity, walls are zero, lids move.
    # y0 (bottom): 0,0 ; x0 (left): 0,0 ; x1 (right): 0,-0.6 ; y1 (top): 1,0
    bc_y0 = make_bc(lambda x: np.isclose(x[1], 0.0), [0.0, 0.0])
    bc_x0 = make_bc(lambda x: np.isclose(x[0], 0.0), [0.0, 0.0])
    bc_x1 = make_bc(lambda x: np.isclose(x[0], 1.0), [0.0, -0.6])
    bc_y1 = make_bc(lambda x: np.isclose(x[1], 1.0), [1.0, 0.0])

    # Pressure pin at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q); p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_y0, bc_x0, bc_x1, bc_y1, bc_p]

    # Stokes initial guess
    w = fem.Function(W)
    (uT, pT) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a_stokes = (nu * ufl.inner(ufl.grad(uT), ufl.grad(v)) * ufl.dx
                - pT * ufl.div(v) * ufl.dx
                + ufl.div(uT) * q * ufl.dx)
    f_const = fem.Constant(msh, np.zeros(gdim))
    L_stokes = ufl.inner(f_const, v) * ufl.dx

    stokes_prob = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_prob.solve()
    w.x.array[:] = w_stokes.x.array[:]

    # Newton NS
    (u, p) = ufl.split(w)
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_const, v) * ufl.dx)
    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    w = problem.solve()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    pts_on, cells_on, idx_map = [], [], []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells_on.append(links[0])
            idx_map.append(i)

    vals = u_sol.eval(np.array(pts_on), np.array(cells_on, dtype=np.int32))
    mag = np.full(pts.shape[0], np.nan)
    mag[idx_map] = np.linalg.norm(vals, axis=1)
    u_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": 0,
            "nonlinear_iterations": [10],
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(spec)
    print("time:", time.time() - t0, "shape:", out["u"].shape, "max:", np.nanmax(out["u"]))
