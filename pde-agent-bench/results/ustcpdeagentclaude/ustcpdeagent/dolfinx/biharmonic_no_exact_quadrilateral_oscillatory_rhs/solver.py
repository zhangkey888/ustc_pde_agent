import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    grid = case_spec["output"]["grid"]
    nx_o, ny_o = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 192
    deg = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [np.array([0.0,0.0]), np.array([1.0,1.0])],
                                 [N,N], cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", deg))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = fem.Function(V); zero.x.array[:] = 0.0
    bc = fem.dirichletbc(zero, bdofs)

    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(8*ufl.pi*x[0])*ufl.cos(6*ufl.pi*x[1])

    # Solve -Δw = f, w=0 on ∂Ω (auxiliary), then -Δu = w, u=0
    u_t = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u_t), ufl.grad(v))*ufl.dx
    L1 = f_expr * v * ufl.dx

    opts = {"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10}
    p1 = petsc.LinearProblem(a, L1, bcs=[bc], petsc_options=opts, petsc_options_prefix="bh1_")
    w = p1.solve()
    it1 = p1.solver.getIterationNumber()

    L2 = ufl.inner(w, v) * ufl.dx
    p2 = petsc.LinearProblem(a, L2, bcs=[bc], petsc_options=opts, petsc_options_prefix="bh2_")
    u = p2.solve()
    it2 = p2.solver.getIterationNumber()

    # Sample
    xs = np.linspace(bbox[0], bbox[1], nx_o)
    ys = np.linspace(bbox[2], bbox[3], ny_o)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_o*ny_o)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []; pproc = []; idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l)>0:
            cells.append(l[0]); pproc.append(pts[i]); idx.append(i)
    vals = u.eval(np.array(pproc), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx_o*ny_o)
    out[idx] = vals
    out = out.reshape(ny_o, nx_o)

    return {"u": out, "solver_info": {
        "mesh_resolution": N, "element_degree": deg,
        "ksp_type":"cg","pc_type":"hypre","rtol":1e-10,
        "iterations": int(it1+it2)
    }}

if __name__ == "__main__":
    import time
    spec = {"output":{"grid":{"nx":128,"ny":128,"bbox":[0,1,0,1]}}}
    t0=time.time(); r=solve(spec); print("time",time.time()-t0)
    print(r["u"].shape, r["u"].min(), r["u"].max())
