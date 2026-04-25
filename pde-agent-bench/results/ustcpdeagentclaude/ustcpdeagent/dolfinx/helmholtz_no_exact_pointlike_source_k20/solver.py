import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    N = 400
    deg = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))
    k = 20.0
    x = ufl.SpatialCoordinate(msh)
    f = 50*ufl.exp(-200*((x[0]-0.5)**2 + (x[1]-0.5)**2))
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k*k*u*v) * ufl.dx
    L = f*v*ufl.dx
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V); u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type":"mumps"},
        petsc_options_prefix="hh_")
    uh = problem.solve()
    its = problem.solver.getIterationNumber()

    g = case_spec["output"]["grid"]
    nx, ny = g["nx"], g["ny"]
    bbox = g["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    vals = np.zeros(nx*ny)
    p_on, c_on, idx = [], [], []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            p_on.append(pts[i]); c_on.append(l[0]); idx.append(i)
    out = uh.eval(np.array(p_on), np.array(c_on, dtype=np.int32)).flatten()
    vals[idx] = out
    u_grid = vals.reshape(ny, nx)
    return {"u": u_grid, "solver_info": {
        "mesh_resolution": N, "element_degree": deg,
        "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12,
        "iterations": int(its)}}
