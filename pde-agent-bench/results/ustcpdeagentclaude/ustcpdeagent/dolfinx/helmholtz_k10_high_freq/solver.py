import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    k = 10.0
    N = 160
    deg = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(3*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    f = -ufl.div(ufl.grad(u_ex)) - k*k*u_ex

    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k*k*u*v)*ufl.dx
    L = f*v*ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type":"preonly","pc_type":"lu"},
        petsc_options_prefix="helm_")
    uh = problem.solve()
    its = problem.solver.getIterationNumber()

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
    for i in range(pts.shape[0]):
        links = coll.links(i)
        cells.append(links[0] if len(links)>0 else 0)
    vals = uh.eval(pts, np.array(cells, dtype=np.int32)).reshape(ny, nx)

    # accuracy check
    exact = np.sin(3*np.pi*XX)*np.sin(2*np.pi*YY)
    err = np.sqrt(np.mean((vals-exact)**2))
    print("RMSE:", err)

    return {"u": vals, "solver_info": {
        "mesh_resolution": N, "element_degree": deg,
        "ksp_type":"preonly","pc_type":"lu","rtol":1e-10,
        "iterations": int(its)
    }}

if __name__ == "__main__":
    import time
    spec = {"output":{"grid":{"nx":128,"ny":128,"bbox":[0,1,0,1]}}}
    t=time.time()
    r = solve(spec)
    print("time:", time.time()-t)
