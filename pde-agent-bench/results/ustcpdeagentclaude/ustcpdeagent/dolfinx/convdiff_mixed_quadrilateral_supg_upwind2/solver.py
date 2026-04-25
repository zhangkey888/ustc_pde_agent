import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    eps_val = 0.005
    beta_val = np.array([18.0, 6.0])
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 128
    degree = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    beta = fem.Constant(msh, PETSc.ScalarType((18.0, 6.0)))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    f = -eps_val*(-2*ufl.pi**2*ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])) \
        + 18.0*ufl.pi*ufl.cos(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) \
        + 6.0*ufl.pi*ufl.sin(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (eps_c*ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(ufl.dot(beta, ufl.grad(u)), v))*ufl.dx
    L = f*v*ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = bnorm*h/(2*eps_c)
    tau = (h/(2*bnorm))*(1.0/ufl.tanh(Pe) - 1.0/Pe)

    # residual (strong form): -eps*lap(u) + beta.grad(u) - f
    if degree >= 2:
        r = -eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    else:
        r = ufl.dot(beta, ufl.grad(u)) - f
    a_supg = tau*ufl.inner(r - (-eps_c*ufl.div(ufl.grad(u)) if degree>=2 else 0)*0, ufl.dot(beta, ufl.grad(v)))*ufl.dx
    # cleaner:
    a_form = a + tau*ufl.inner(-eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v)))*ufl.dx if degree>=2 else a + tau*ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v)))*ufl.dx
    L_form = L + tau*f*ufl.dot(beta, ufl.grad(v))*ufl.dx

    # BC
    fdim = msh.topology.dim - 1
    bf = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, bf))

    problem = petsc.LinearProblem(a_form, L_form, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="cd_")
    uh = problem.solve()

    its = problem.solver.getIterationNumber()

    # Sample
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    pp = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            pp.append(pts[i]); cells.append(l[0]); idx.append(i)
    vals = uh.eval(np.array(pp), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx_out*ny_out)
    out[idx] = vals
    u_grid = out.reshape(ny_out, nx_out)

    return {"u": u_grid, "solver_info": {
        "mesh_resolution": N, "element_degree": degree,
        "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12,
        "iterations": int(its)
    }}
