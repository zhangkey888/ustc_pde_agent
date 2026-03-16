import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Manufactured solution: u = x*(1-x)*y*(1-y)
    # Laplacian: -2*y*(1-y) - 2*x*(1-x)
    # Biharmonic (Laplacian of Laplacian): -2*(-2) - 2*(-2) = 4+4 = 8? Let me compute carefully.
    # u = x(1-x)y(1-y) = (x - x^2)(y - y^2)
    # u_xx = -2*y*(1-y)
    # u_yy = -2*x*(1-x)
    # Δu = -2y(1-y) - 2x(1-x)
    # (Δu)_xx = -2*(-2) = 4... wait:
    # Δu = -2y + 2y^2 - 2x + 2x^2
    # (Δu)_xx = 4
    # (Δu)_yy = 4
    # Δ²u = 4 + 4 = 8
    # So f = 8
    
    # Mixed formulation: introduce w = -Δu
    # -Δu = w  =>  Δw = -f  (since Δ²u = f means Δ(Δu) = f, so Δw = -f)
    # Actually: Δ²u = Δ(Δu) = f
    # Let w = Δu, then Δw = f
    # But we solve: -Δu = -w (weak form with negative sign)
    # 
    # Better: Let w = -Δu (so w satisfies -Δu = w)
    # Then Δ²u = Δ(Δu) = -Δw = f => -Δw = f
    #
    # Step 1: Solve -Δw = f with w = -Δu_exact on boundary
    # Step 2: Solve -Δu = w with u = g on boundary
    
    # w_exact = -Δu_exact = 2y(1-y) + 2x(1-x)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_val = fem.Constant(domain, PETSc.ScalarType(8.0))
    
    # Step 1: Solve -Δw = f, w = w_exact on ∂Ω
    # w_exact = 2*y*(1-y) + 2*x*(1-x)
    # On boundary of unit square, at least one of x or y is 0 or 1
    # If x=0: w = 2y(1-y); x=1: w = 2y(1-y); y=0: w = 2x(1-x); y=1: w = 2x(1-x)
    
    w_func = fem.Function(V)
    u_func = fem.Function(V)
    
    # BC for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # w boundary condition
    w_bc_func = fem.Function(V)
    w_bc_func.interpolate(lambda x: 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]))
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    # Solve -Δw = f
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_val, v_test) * ufl.dx
    
    total_iterations = 0
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_",
    )
    w_func = problem1.solve()
    
    # Get iteration count
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w, u = 0 on ∂Ω (zero BC)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.zeros_like(x[0]))
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_func, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_",
    )
    u_func = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # Evaluate on 50x50 grid
    n_pts = 50
    xs = np.linspace(0, 1, n_pts)
    ys = np.linspace(0, 1, n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_pts * n_pts))
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
    
    u_values = np.full(n_pts * n_pts, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_pts, n_pts))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        },
    }