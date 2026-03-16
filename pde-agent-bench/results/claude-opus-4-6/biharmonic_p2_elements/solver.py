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
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  Δu = w,  Δw = f
    # with u = g on ∂Ω, and w = Δu on ∂Ω (from manufactured solution)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution
    u_exact_ufl = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Laplacian of u_exact:
    # Δu = -4π²sin(2πx)sin(2πy) - 4π²sin(2πx)sin(2πy) = -8π²sin(2πx)sin(2πy)
    w_exact_ufl = -8 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # f = Δw = Δ(Δu) = 64π⁴ sin(2πx)sin(2πy)
    f_ufl = 64 * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    
    # Step 1: Solve -Δw = -f with w = w_exact on ∂Ω
    # Actually: Δw = f, so weak form: -∫∇w·∇v dx = -∫f v dx (with integration by parts of Δw)
    # Standard: ∫∇w·∇v dx = -∫f v dx ... no.
    # Let's be careful. We have Δw = f.
    # Multiply by test function v, integrate: ∫Δw v dx = ∫f v dx
    # Integration by parts: -∫∇w·∇v dx + ∫(∂w/∂n)v ds = ∫f v dx
    # With Dirichlet BCs on w (boundary terms handled by BCs):
    # -∫∇w·∇v dx = ∫f v dx  =>  ∫∇w·∇v dx = -∫f v dx
    # 
    # Hmm, that gives a negative RHS. Let me use the standard approach:
    # Solve: -Δw = -f  =>  ∫∇w·∇v dx = ∫f v dx ... no that's -Δw = -f => ∫∇w·∇v = ∫(-f)v ... 
    #
    # Let me just be systematic:
    # Equation: Δw = f
    # Weak form (multiply by v, integrate by parts once):
    #   -∫∇w·∇v dx = ∫f v dx - boundary terms
    # So: ∫∇w·∇v dx = -∫f v dx (with Dirichlet BCs absorbing boundary terms)
    #
    # This means the bilinear form is positive definite but the RHS is negative of ∫fv.
    # Actually the bilinear form ∫∇w·∇v dx is positive definite, and we need:
    # ∫∇w·∇v dx = -∫f v dx
    #
    # Alternatively, define σ = -Δu, then -Δσ = -f, i.e., -Δσ = -f => ∫∇σ·∇v = -∫f v ... same issue.
    #
    # Let me use a different splitting. Define w = -Δu (note the sign).
    # Then -Δu = -w => ∫∇u·∇v = ∫w v (standard Poisson)
    # And Δ²u = f => Δ(-w) = f => -Δw = f => ∫∇w·∇v = ∫f v (standard Poisson)
    #
    # So with w = -Δu:
    # Step 1: Solve ∫∇w·∇v dx = ∫f v dx, with w = -Δu_exact = 8π²sin(2πx)sin(2πy) on ∂Ω (which is 0)
    # Step 2: Solve ∫∇u·∇v dx = ∫w v dx, with u = u_exact on ∂Ω (which is 0)
    
    # w_bc_exact = -Δu_exact = 8π²sin(2πx)sin(2πy) = 0 on boundary (since sin(0)=sin(2π)=0)
    # u_bc_exact = sin(2πx)sin(2πy) = 0 on boundary
    
    # So both have zero Dirichlet BCs! Great.
    
    total_iterations = 0
    
    # Step 1: Solve for w where -Δw = f, i.e., ∫∇w·∇v dx = ∫f v dx
    w_sol = fem.Function(V)
    v_test = ufl.TestFunction(V)
    w_trial = ufl.TrialFunction(V)
    
    # BC for w: w = 0 on boundary (since w = -Δu_exact = 8π²sin(2πx)sin(2πy) = 0 on ∂Ω)
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(ScalarType(0.0), dofs_w, V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_ufl, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem1.solve()
    
    # Get iteration count from step 1
    ksp1 = problem1.solver
    iter1 = ksp1.getIterationNumber()
    total_iterations += iter1
    
    # Step 2: Solve for u where -Δu = w, i.e., ∫∇u·∇v dx = ∫w v dx
    u_trial = ufl.TrialFunction(V)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(ScalarType(0.0), dofs_u, V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    iter2 = ksp2.getIterationNumber()
    total_iterations += iter2
    
    # Evaluate on 50x50 grid
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
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }