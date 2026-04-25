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

    eps_val = 0.01
    beta_val = np.array([12.0, 0.0])

    N = 64
    degree = 3
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(3*x[0]) * ufl.sin(ufl.pi * x[1])
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))

    # f = -eps*lap(u) + beta·grad(u)
    lap = ufl.div(ufl.grad(u_exact))
    f = -eps_c * lap + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)*ufl.dx
    L = f*v*ufl.dx

    # SUPG
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(bnorm*h/(2*eps_c)) - 2*eps_c/(bnorm*h))
    R = -eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    a += tau * ufl.dot(beta, ufl.grad(v)) * (-eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    # BC
    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, bfacets))

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                                   petsc_options_prefix="cd_")
    uh = problem.solve()

    # Sample
    xs = np.linspace(bbox[0], bbox[1], nx_o)
    ys = np.linspace(bbox[2], bbox[3], ny_o)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_o*ny_o)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        cells.append(l[0] if len(l)>0 else 0)
    vals = uh.eval(pts, np.array(cells, dtype=np.int32)).reshape(ny_o, nx_o)

    return {"u": vals, "solver_info": {
        "mesh_resolution": N, "element_degree": degree,
        "ksp_type":"preonly","pc_type":"lu","rtol":1e-12,"iterations":1
    }}
