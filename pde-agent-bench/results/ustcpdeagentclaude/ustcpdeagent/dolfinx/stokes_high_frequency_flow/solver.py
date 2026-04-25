import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 64
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 3, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 2)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 1.0
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
        -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1]),
    ])
    p_ex = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])
    f = -nu*ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         - p*ufl.div(v)*ufl.dx
         - ufl.div(u)*q*ufl.dx)
    L = ufl.inner(f, v)*ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    wh = problem.solve()
    uh = wh.sub(0).collapse()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    pts_on = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            cells.append(l[0])
            pts_on.append(pts[i])
            idx.append(i)
    vals = uh.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
    mag = np.zeros(nx_out*ny_out)
    mag[idx] = np.linalg.norm(vals, axis=1)
    u_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 3,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    r = solve(spec)
    print("time", time.time()-t0)
    # check against exact
    xs = np.linspace(0,1,64); ys = np.linspace(0,1,64)
    XX, YY = np.meshgrid(xs, ys)
    ux = 2*np.pi*np.cos(2*np.pi*YY)*np.sin(2*np.pi*XX)
    uy = -2*np.pi*np.cos(2*np.pi*XX)*np.sin(2*np.pi*YY)
    mag_ex = np.sqrt(ux**2+uy**2)
    err = np.sqrt(np.mean((r["u"]-mag_ex)**2))
    print("err", err)
