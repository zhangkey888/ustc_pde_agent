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

    eps_val = 0.01
    beta_val = np.array([15.0, 0.0])

    N = 96
    degree = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))
    f = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_f = f
    a += tau * ufl.inner(R_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(R_f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    # BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_")
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    ppts = []
    idx = []
    for i in range(pts.shape[0]):
        links = col.links(i)
        if len(links) > 0:
            cells.append(links[0]); ppts.append(pts[i]); idx.append(i)
    vals = np.full(pts.shape[0], np.nan)
    v_arr = u_sol.eval(np.array(ppts), np.array(cells, dtype=np.int32)).flatten()
    vals[idx] = v_arr
    u_grid = vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type": "gmres", "pc_type": "ilu", "rtol": 1e-10,
            "iterations": int(iters),
        }
    }
