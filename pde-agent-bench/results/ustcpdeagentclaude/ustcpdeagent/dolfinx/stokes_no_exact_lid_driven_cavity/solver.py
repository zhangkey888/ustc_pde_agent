import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    nu_val = 0.2
    N = 280
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # Top lid (y=1): u = (1, 0). Use a smoothly-capped profile? No, use exact as spec.
    # To avoid corner singularity issues in discrete sense, still use (1,0) on top.
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))

    # Other walls: u = 0
    other_facets = mesh.locate_entities_boundary(
        msh, fdim,
        lambda x: np.isclose(x[1], 0.0) | np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    )
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_other = fem.locate_dofs_topological((W.sub(0), V), fdim, other_facets)
    bc_other = fem.dirichletbc(u_zero, dofs_other, W.sub(0))

    # Pressure pin at corner (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_top, bc_other, bc_p]

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_ldc_"
    )
    w_h = problem.solve()

    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

    u_h = w_h.sub(0).collapse()

    # Sample on grid
    out = case_spec["output"]["grid"]
    nx = out["nx"]; ny = out["ny"]
    bbox = out["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    cells = []
    pts_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if len(pts_on_proc) > 0:
        vals = u_h.eval(np.array(pts_on_proc), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals

    mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    r = solve(spec)
    print("Time:", time.time() - t0)
    print("Shape:", r["u"].shape, "max:", r["u"].max(), "min:", r["u"].min())
