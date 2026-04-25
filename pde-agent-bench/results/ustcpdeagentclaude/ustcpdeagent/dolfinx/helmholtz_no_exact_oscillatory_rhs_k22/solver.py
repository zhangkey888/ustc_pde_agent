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
    k = 22.0

    N = 160
    degree = 3
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V); u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.sin(10*ufl.pi*x[0]) * ufl.sin(8*ufl.pi*x[1])
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k*k*ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type":"preonly","pc_type":"lu",
                                                  "pc_factor_mat_solver_type":"mumps"},
                                   petsc_options_prefix="helm_")
    u_sol = problem.solve()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []; pts_ok = []; idx = []
    for i in range(pts.shape[0]):
        L_i = coll.links(i)
        if len(L_i) > 0:
            cells.append(L_i[0]); pts_ok.append(pts[i]); idx.append(i)
    vals = u_sol.eval(np.array(pts_ok), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx_out*ny_out)
    out[idx] = vals
    u_grid = out.reshape(ny_out, nx_out)

    return {"u": u_grid, "solver_info": {
        "mesh_resolution": N, "element_degree": degree,
        "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12,
        "iterations": 1
    }}
