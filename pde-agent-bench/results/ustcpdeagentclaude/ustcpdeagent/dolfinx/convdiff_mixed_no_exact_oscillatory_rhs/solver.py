import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.005
    beta_vals = [15.0, 7.0]

    # Mesh resolution — high Pe, need fine mesh
    N = 640
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])

    beta = fem.Constant(msh, PETSc.ScalarType((beta_vals[0], beta_vals[1])))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps))

    # Galerkin form
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    # Pe_h = |beta|*h/(2*eps)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau for SUPG
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Residual (strong form): -eps*div(grad(u)) + beta.grad(u) - f
    # For P1, laplacian is 0 on triangles
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_L = f_expr

    a_supg = tau * R_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_supg = tau * R_L * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    # Dirichlet BC u=0 on entire boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))

    rtol = 1e-9
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="cd_",
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample onto uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    cells = []
    pts_list = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_list.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.full(pts.shape[0], np.nan)
    if len(pts_list) > 0:
        vals = u_sol.eval(np.array(pts_list), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"u range: [{np.nanmin(res['u']):.4f}, {np.nanmax(res['u']):.4f}]")
    print(f"iters: {res['solver_info']['iterations']}")
