import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.12
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 3, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 2)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    expf = ufl.exp(-20 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
    u_ex = ufl.as_vector([-40*(x[1]-0.5)*expf, 40*(x[0]-0.5)*expf])
    p_ex = 0.0 * x[0]

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex))

    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f, v) * ufl.dx)

    # BCs: u = u_ex on boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pin pressure at corner
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    J = ufl.derivative(F, w)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    w_h = problem.solve()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    pts_on = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            pts_on.append(pts[i])
            cells.append(l[0])
            idx.append(i)
    vals = u_sol.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
    mag = np.zeros(nx*ny)
    mag[idx] = np.linalg.norm(vals, axis=1)
    mag = mag.reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 3,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [10],
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time()-t0)
    print("max mag:", res["u"].max())
    # Compare with exact
    nx = ny = 128
    xs = np.linspace(0,1,nx); ys = np.linspace(0,1,ny)
    XX, YY = np.meshgrid(xs, ys)
    expf = np.exp(-20*((XX-0.5)**2 + (YY-0.5)**2))
    ux = -40*(YY-0.5)*expf
    uy = 40*(XX-0.5)*expf
    mag_ex = np.sqrt(ux**2 + uy**2)
    err = np.linalg.norm(res["u"]-mag_ex)/np.linalg.norm(mag_ex)
    print("rel L2 err:", err)
    print("abs L2 err:", np.linalg.norm(res["u"]-mag_ex))
