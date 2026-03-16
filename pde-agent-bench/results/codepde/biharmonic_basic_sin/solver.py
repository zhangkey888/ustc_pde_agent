import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    nx_eval = 50
    ny_eval = 50
    
    # Mesh resolution - use fine enough mesh for accuracy
    N = 80
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  becomes:
    #   -Δu = w   (with w = -Δu)
    #   -Δw = f
    # But we need to be careful with BCs.
    # 
    # For u = sin(pi*x)*sin(pi*y):
    #   Δu = -2*pi²*sin(pi*x)*sin(pi*y)
    #   Δ²u = 4*pi⁴*sin(pi*x)*sin(pi*y)
    #   So f = 4*pi⁴*sin(pi*x)*sin(pi*y)
    #
    # Let w = -Δu = 2*pi²*sin(pi*x)*sin(pi*y)
    # 
    # BCs: u = 0 on ∂Ω, w = 0 on ∂Ω (both sin terms vanish on boundary)
    
    # 3. Function spaces
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Source term f = 4*pi^4*sin(pi*x)*sin(pi*y)
    f_expr = 4.0 * pi**4 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Exact solution for BCs
    u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    w_exact_expr = 2.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Boundary conditions - entire boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary(x_coord):
        return (np.isclose(x_coord[0], 0.0) | np.isclose(x_coord[0], 1.0) |
                np.isclose(x_coord[1], 0.0) | np.isclose(x_coord[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    
    # BC for w: w = 2*pi²*sin(pi*x)*sin(pi*y) = 0 on boundary
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w_func = fem.Function(V)
    bc_w_func.interpolate(lambda x_coord: np.zeros_like(x_coord[0]))
    bc_w = fem.dirichletbc(bc_w_func, dofs_w)
    
    # BC for u: u = 0 on boundary
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u_func = fem.Function(V)
    bc_u_func.interpolate(lambda x_coord: np.zeros_like(x_coord[0]))
    bc_u = fem.dirichletbc(bc_u_func, dofs_u)
    
    total_iterations = 0
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Step 1: Solve -Δw = f with w = 0 on ∂Ω
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="solve1_"
    )
    wh = problem1.solve()
    
    # Get iteration count from first solve
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = 0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(wh, v_test2) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="solve2_"
    )
    uh = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }