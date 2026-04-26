import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Parameters - P3 with mesh=96 gives excellent accuracy within time budget
    mesh_res = 96
    elem_degree = 3
    rtol = 1e-10

    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Source term: f = Δ²u
    # For u = sin(πx)sin(πy) + 0.5*sin(2πx)sin(3πy):
    # Δ²u = 4π⁴sin(πx)sin(πy) + (169/2)π⁴sin(2πx)sin(3πy)
    f_expr = (4.0 * pi**4 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + (169.0 / 2.0) * pi**4 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1]))

    # Boundary DOFs (homogeneous Dirichlet since u=0 on ∂Ω)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # ---- Step 1: Solve -Δσ = -f, σ=0 on ∂Ω (since σ = Δu = -2π²sin(πx)sin(πy) - (13/2)π²sin(2πx)sin(3πy) = 0 on ∂Ω) ----
    sigma_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a1 = ufl.inner(ufl.grad(sigma_trial), ufl.grad(v)) * ufl.dx
    L1 = -ufl.inner(f_expr, v) * ufl.dx

    bc_sigma = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    problem1 = petsc.LinearProblem(a1, L1, bcs=[bc_sigma],
                                    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
                                    petsc_options_prefix="biharm1_")
    sigma_sol = problem1.solve()
    sigma_sol.x.scatter_forward()

    ksp1 = problem1.solver
    iterations1 = ksp1.getIterationNumber()

    # ---- Step 2: Solve -Δu = -σ, u=0 on ∂Ω ----
    u_trial = ufl.TrialFunction(V)

    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L2 = -ufl.inner(sigma_sol, v) * ufl.dx

    bc_u = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    problem2 = petsc.LinearProblem(a2, L2, bcs=[bc_u],
                                    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
                                    petsc_options_prefix="biharm2_")
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()

    ksp2 = problem2.solver
    iterations2 = ksp2.getIterationNumber()

    total_iterations = iterations1 + iterations2

    # ---- Evaluate on output grid ----
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    # Gather in parallel
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # ---- Compute L2 error for verification ----
    u_exact_expr = (ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
                    + 0.5 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1]))
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    L2_error_sq = fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx))
    L2_error = np.sqrt(domain.comm.allreduce(L2_error_sq, op=MPI.SUM))
    print(f"L2 error: {L2_error:.6e}")
    print(f"Total KSP iterations: {total_iterations}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
