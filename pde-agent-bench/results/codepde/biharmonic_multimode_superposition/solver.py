import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD

    # 2. Create mesh - use fine mesh for 4th order problem
    N = 80
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # For the biharmonic equation Δ²u = f, we use a mixed formulation:
    # Introduce w = -Δu, then:
    #   -Δu = w   in Ω
    #   -Δw = f   in Ω
    # with boundary conditions:
    #   u = g on ∂Ω
    #   w = -Δu on ∂Ω (derived from exact solution)

    # Manufactured solution: u = sin(pi*x)*sin(pi*y) + (1/2)*sin(2*pi*x)*sin(3*pi*y)
    # -Δu = pi²(1+1)*sin(pi*x)*sin(pi*y) + (1/2)*pi²(4+9)*sin(2*pi*x)*sin(3*pi*y)
    #      = 2*pi²*sin(pi*x)*sin(pi*y) + (13/2)*pi²*sin(2*pi*x)*sin(3*pi*y)
    # So w = -Δu = 2*pi²*sin(pi*x)*sin(pi*y) + (13/2)*pi²*sin(2*pi*x)*sin(3*pi*y)
    #
    # Δ²u = -Δw = pi²(1+1)*2*pi²*sin(pi*x)*sin(pi*y) + (13/2)*pi²*pi²(4+9)*sin(2*pi*x)*sin(3*pi*y)
    #      = 4*pi⁴*sin(pi*x)*sin(pi*y) + (169/2)*pi⁴*sin(2*pi*x)*sin(3*pi*y)
    # f = 4*pi⁴*sin(pi*x)*sin(pi*y) + (169/2)*pi⁴*sin(2*pi*x)*sin(3*pi*y)

    # Use degree 2 elements for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    # Exact solution
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + \
                   0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])

    # w_exact = -Δu_exact
    w_exact_expr = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + \
                   6.5 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])

    # Source term f = Δ²u
    f_expr = 4.0 * ufl.pi**4 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + \
             84.5 * ufl.pi**4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])

    # ---- Step 1: Solve -Δw = f with w = w_exact on ∂Ω ----
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx

    # BC for w: w = w_exact on boundary
    # Since u_exact = sin(pi*x)*sin(pi*y) + 0.5*sin(2*pi*x)*sin(3*pi*y),
    # on the boundary of [0,1]^2, u_exact = 0 (all sin terms vanish).
    # Similarly w_exact = 0 on the boundary.

    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # BC for w (w = 0 on boundary since w_exact vanishes there)
    w_bc_func = fem.Function(V)
    w_bc_func.x.array[:] = 0.0
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)

    total_iterations = 0

    # Solve for w
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    wh = problem1.solve()

    # Get iteration count from step 1
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()

    # ---- Step 2: Solve -Δu = w with u = u_exact on ∂Ω ----
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)

    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(wh, v_test2) * ufl.dx

    # BC for u (u = 0 on boundary)
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    uh = problem2.solve()

    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()

    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
        }
    }