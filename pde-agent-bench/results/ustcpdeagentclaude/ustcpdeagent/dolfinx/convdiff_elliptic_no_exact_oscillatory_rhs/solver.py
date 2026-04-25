import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps_val = 0.05
    beta_val = np.array([3.0, 3.0])

    # Grid output spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh resolution and element degree
    N = 256
    deg = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", deg))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])

    epsilon = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))

    # Standard Galerkin
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    # Péclet number local
    Pe_h = beta_norm * h / (2 * epsilon)
    # tau standard
    tau = (h / (2 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Residual for SUPG: strong residual of equation
    # -eps*div(grad(u)) + beta.grad(u) - f
    r_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_L = f_expr

    a += tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(r_L, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    # Dirichlet BC u=0 on all boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre",
                       "ksp_rtol": rtol, "ksp_atol": 1e-14,
                       "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample onto uniform grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Shape: {result['u'].shape}")
    print(f"Min/Max: {result['u'].min():.4e} / {result['u'].max():.4e}")
    print(f"Info: {result['solver_info']}")
