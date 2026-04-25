import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    N = 64
    deg = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(3*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    kappa = 1 + 0.9*ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    f = -ufl.div(kappa*ufl.grad(u_ex))

    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = f*v*ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10},
        petsc_options_prefix="pois_")
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
    col = geometry.compute_colliding_cells(msh, cand, pts)
    cells = np.array([col.links(i)[0] for i in range(len(pts))], dtype=np.int32)
    vals = uh.eval(pts, cells).reshape(ny, nx)

    return {"u": vals, "solver_info": {
        "mesh_resolution": N, "element_degree": deg,
        "ksp_type":"cg","pc_type":"hypre","rtol":1e-10,"iterations":int(its)
    }}
