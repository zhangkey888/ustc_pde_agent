import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 80
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create quadrilateral mesh as specified by case ID
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # Manufactured solution: u = sin(2*pi*x)*cos(3*pi*y)
    # Laplacian: -(4pi^2 + 9pi^2)*sin(2*pi*x)*cos(3*pi*y) = -13*pi^2 * u_exact
    # Bilaplacian = Laplacian of Laplacian:
    # Lap(u) = -13*pi^2 * sin(2*pi*x)*cos(3*pi*y)
    # Lap(Lap(u)) = -13*pi^2 * Lap(sin(2*pi*x)*cos(3*pi*y)) = -13*pi^2 * (-13*pi^2) * sin(2*pi*x)*cos(3*pi*y)
    # = 169*pi^4 * sin(2*pi*x)*cos(3*pi*y)
    
    # Mixed formulation: introduce w = -Lap(u)
    # -Lap(u) = w  =>  Lap(u) + w = 0
    # -Lap(w) = f  =>  Lap(w) + f = 0
    # So we solve two Poisson problems:
    # Step 1: -Lap(w) = f with w = -Lap(u_exact) on boundary
    # Step 2: -Lap(u) = w with u = u_exact on boundary
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    u_exact_expr = ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    # w_exact = -Lap(u_exact) = 13*pi^2 * sin(2*pi*x)*cos(3*pi*y)
    w_exact_expr = 13.0 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    # f = Lap^2(u) = 169*pi^4 * sin(2*pi*x)*cos(3*pi*y)
    f_expr = 169.0 * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    total_iterations = 0
    
    # Step 1: Solve -Lap(w) = f, w = w_exact on boundary
    w_bc = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc.interpolate(w_bc_expr)
    bc_w = fem.dirichletbc(w_bc, boundary_dofs)
    
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    prefix1 = "step1_"
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix=prefix1,
    )
    w_sol = problem1.solve()
    
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Lap(u) = w, u = u_exact on boundary
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs)
    
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test2) * ufl.dx
    
    prefix2 = "step2_"
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix=prefix2,
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
        },
    }