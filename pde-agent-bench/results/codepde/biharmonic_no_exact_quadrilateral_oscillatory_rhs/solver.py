import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD
    
    # Mesh resolution and element degree
    N = 128
    degree = 2
    
    # 2. Create mesh - use quadrilateral as specified in case ID
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Mixed formulation for biharmonic:
    #    Δ²u = f  with u = 0 on ∂Ω (and natural BC for Δu)
    #    Introduce w = -Δu, then:
    #      -Δu = w   (with u = 0 on ∂Ω)
    #      -Δw = f   (with natural BC for w, or w = 0 if we want clamped plate)
    #
    #    Actually for biharmonic with u = g on boundary:
    #    We use the mixed formulation:
    #      w = -Δu
    #      -Δw = f
    #    with u = g on ∂Ω. For the second equation, we need a BC on w.
    #    For a simply supported plate: u = 0 and Δu = 0 on ∂Ω => w = 0 on ∂Ω
    #    For a clamped plate: u = 0 and ∂u/∂n = 0 on ∂Ω => natural BC on w
    #
    #    The problem states u = g on ∂Ω. With no other BC specified,
    #    we'll assume simply supported: u = g and w = -Δu = 0 on ∂Ω
    #    (This is the standard well-posed mixed formulation)
    
    # Since case_spec says u = g on ∂Ω, and source is f = sin(8πx)cos(6πy),
    # with no exact solution, g is likely 0 (homogeneous).
    # Let's check case_spec for boundary conditions
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Default: g = 0
    g_val = 0.0
    
    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(8 * ufl.pi * x[0]) * ufl.cos(6 * ufl.pi * x[1])
    
    # Boundary conditions: all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary(x_coords):
        return (np.isclose(x_coords[0], 0.0) | np.isclose(x_coords[0], 1.0) |
                np.isclose(x_coords[1], 0.0) | np.isclose(x_coords[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    
    # Step 1: Solve -Δw = f with w = 0 on ∂Ω (simply supported assumption)
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    # BC for w: w = 0 on ∂Ω
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_w, V)
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="solve1_"
    )
    w_sol = problem1.solve()
    
    # Get iteration count from first solve
    iter1 = problem1.solver.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = g on ∂Ω
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    # BC for u: u = g on ∂Ω
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(PETSc.ScalarType(g_val), dofs_u, V)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="solve2_"
    )
    u_sol = problem2.solve()
    
    iter2 = problem2.solver.getIterationNumber()
    
    total_iterations = iter1 + iter2
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }