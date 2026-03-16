import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    pi = ufl.pi
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + \
                   0.5 * ufl.sin(2 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # For the biharmonic equation Δ²u = f, we use a mixed formulation:
    # Introduce w = -Δu, then:
    #   -Δu = w  in Ω
    #   -Δw = f  in Ω
    # with appropriate boundary conditions.
    
    # Compute Laplacian of exact solution analytically:
    # u = sin(πx)sin(πy) + 0.5*sin(2πx)sin(3πy)
    # Δu = -π²(1+1)sin(πx)sin(πy) + 0.5*(-π²(4+9))sin(2πx)sin(3πy)
    #     = -2π²sin(πx)sin(πy) - 6.5π²sin(2πx)sin(3πy)
    # So w = -Δu = 2π²sin(πx)sin(πy) + 6.5π²sin(2πx)sin(3πy)
    
    # Δ²u = Δ(Δu) = Δ(-2π²sin(πx)sin(πy) - 6.5π²sin(2πx)sin(3πy))
    #      = -2π²*(-2π²)sin(πx)sin(πy) - 6.5π²*(-13π²)sin(2πx)sin(3πy)
    #      = 4π⁴sin(πx)sin(πy) + 84.5π⁴sin(2πx)sin(3πy)
    #      = 4π⁴sin(πx)sin(πy) + (169/2)π⁴sin(2πx)sin(3πy)
    
    f_ufl = 4 * pi**4 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + \
            (169.0 / 2.0) * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    w_exact_ufl = 2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + \
                  6.5 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    # Since the exact solution has sin terms that vanish on the boundary of [0,1]²,
    # w_exact = 0 on ∂Ω (all sin(nπ*0) = 0 and sin(nπ*1) = 0)
    # Similarly u_exact = 0 on ∂Ω
    
    # Boundary conditions for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # w boundary condition (w = w_exact on ∂Ω, which is 0)
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_ufl, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    bc_w = fem.dirichletbc(w_bc_func, boundary_dofs)
    
    # Solve for w: -Δw = f => ∫∇w·∇v dx = ∫f·v dx
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_form = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_form_w = ufl.inner(f_ufl, v_test) * ufl.dx
    
    problem_w = petsc.LinearProblem(
        a_form, L_form_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem_w.solve()
    
    # Get iteration count
    ksp1 = problem_w.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = u_exact on ∂Ω (which is 0)
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    u_trial = ufl.TrialFunction(V)
    
    a_form2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L_form_u = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_form2, L_form_u, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem_u.solve()
    
    ksp2 = problem_u.solver
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