import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh - quadrilateral
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.quadrilateral)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  w = -Δu,  -Δw = f  (with appropriate BCs)
    # Or equivalently: -Δu = w, -Δw = f
    # BCs: u = g on ∂Ω, and w = -Δu on ∂Ω (computed from exact solution)
    
    # Manufactured solution: u = sin(2πx)*sin(πy)
    # Δu = -(4π² + π²) sin(2πx)sin(πy) = -5π² sin(2πx)sin(πy)
    # w = -Δu = 5π² sin(2πx)sin(πy)
    # Δw = -5π²(4π² + π²) sin(2πx)sin(πy) = -25π⁴ sin(2πx)sin(πy)
    # f = -Δw = 25π⁴ sin(2πx)sin(πy)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    w_exact_expr = 5.0 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = 25.0 * ufl.pi**4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC for w (auxiliary variable): w = -Δu = 5π² sin(2πx)sin(πy)
    w_bc_func = fem.Function(V)
    w_bc_expr_compiled = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr_compiled)
    bc_w = fem.dirichletbc(w_bc_func, boundary_dofs)
    
    # BC for u: u = sin(2πx)sin(πy) = 0 on boundary of unit square
    u_bc_func = fem.Function(V)
    u_bc_expr_compiled = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr_compiled)
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    w_sol = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_sol), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    w_h = problem1.solve()
    
    # Get iteration count
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = u_exact on ∂Ω
    u_sol = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_sol), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_h, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_"
    )
    u_h = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }