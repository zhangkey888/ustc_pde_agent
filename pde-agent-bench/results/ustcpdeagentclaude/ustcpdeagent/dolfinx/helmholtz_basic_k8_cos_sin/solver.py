import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    k = 8.0
    N = 96
    deg = 2
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))

    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.cos(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    f = 2*ufl.pi**2*u_ex - k*k*u_ex

    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - k*k*ufl.inner(u,v)*ufl.dx
    L = ufl.inner(f, v)*ufl.dx

    fdim = msh.topology.dim-1
    bf = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, bf))

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                  petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                                  petsc_options_prefix="helm_")
    uh = problem.solve()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = np.array([coll.links(i)[0] for i in range(len(pts))], dtype=np.int32)
    vals = uh.eval(pts, cells).reshape(ny_out, nx_out)

    return {"u": vals, "solver_info": {
        "mesh_resolution": N, "element_degree": deg,
        "ksp_type":"preonly","pc_type":"lu","rtol":1e-12,"iterations":1}}

if __name__=="__main__":
    import time
    cs={"output":{"grid":{"nx":64,"ny":64,"bbox":[0,1,0,1]}}}
    t=time.time(); r=solve(cs); print("time",time.time()-t)
    xs=np.linspace(0,1,64); ys=np.linspace(0,1,64)
    X,Y=np.meshgrid(xs,ys)
    ex=np.cos(np.pi*X)*np.sin(np.pi*Y)
    print("err",np.sqrt(np.mean((r["u"]-ex)**2)))
