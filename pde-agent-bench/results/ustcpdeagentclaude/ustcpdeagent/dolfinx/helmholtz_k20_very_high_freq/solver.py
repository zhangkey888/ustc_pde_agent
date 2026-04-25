import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    k = 20.0
    N = 160
    deg = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))

    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(6*ufl.pi*x[0]) * ufl.sin(5*ufl.pi*x[1])
    f = -ufl.div(ufl.grad(u_ex)) - k*k * u_ex

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k*k*ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bf = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, bf)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                  "pc_factor_mat_solver_type": "mumps"},
                                   petsc_options_prefix="helm_")
    uh = problem.solve()
    its = 1

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
    pp = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            pp.append(pts[i])
            cells.append(l[0])
            idx.append(i)
    vals = uh.eval(np.array(pp), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx*ny)
    out[idx] = vals
    u_grid = out.reshape(ny, nx)

    return {"u": u_grid,
            "solver_info": {"mesh_resolution": N, "element_degree": deg,
                            "ksp_type": "preonly", "pc_type": "lu",
                            "rtol": 1e-10, "iterations": its}}

if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    r = solve(spec)
    print("time", time.time()-t0)
    xs = np.linspace(0,1,64); ys = np.linspace(0,1,64)
    XX,YY = np.meshgrid(xs,ys)
    ex = np.sin(6*np.pi*XX)*np.sin(5*np.pi*YY)
    print("err", np.sqrt(np.mean((r["u"]-ex)**2)))
