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
    
    # Create quadrilateral mesh on unit square
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δu = w,  -Δw = f
    # with u = g on ∂Ω and w = -Δu_exact on ∂Ω
    
    # Manufactured solution: u = sin(2πx)*cos(3πy)
    # Δu = -(4π² + 9π²) sin(2πx)*cos(3πy) = -13π² sin(2πx)*cos(3πy)
    # w = -Δu = 13π² sin(2πx)*cos(3πy)
    # Δw = 13π² * (-(4π² + 9π²)) sin(2πx)*cos(3πy) = -13²π⁴ sin(2πx)*cos(3πy)
    # f = -Δw = 13²π⁴ sin(2πx)*cos(3πy) = 169π⁴ sin(2πx)*cos(3πy)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_expr = ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    w_exact_expr = 13.0 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    f_expr = 169.0 * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # BC for w (first solve): w = w_exact on ∂Ω
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    # BC for u (second solve): u = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    # First solve: -Δw = f  =>  (∇w, ∇v) = (f, v)
    w_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_expr, v) * ufl.dx
    
    total_iterations = 0
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="solve1_"
    )
    w_sol = problem1.solve()
    total_iterations += problem1.solver.getIterationNumber()
    
    # Second solve: -Δu = w  =>  (∇u, ∇v) = (w, v)
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_sol, v) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="solve2_"
    )
    u_sol = problem2.solve()
    total_iterations += problem2.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
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