import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Parameters
    nx = ny = 100
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  with u = 0 on ∂Ω (and we assume Δu = 0 on ∂Ω for simply supported)
    # 
    # Introduce w = -Δu, then:
    #   -Δu = w  in Ω,  u = 0 on ∂Ω
    #   -Δw = f  in Ω,  w = 0 on ∂Ω  (simply supported plate)
    #
    # We solve in two steps:
    # Step 1: -Δw = f, w=0 on ∂Ω
    # Step 2: -Δu = w, u=0 on ∂Ω
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = cos(4*pi*x)*sin(3*pi*y)
    pi = ufl.pi
    f_expr = ufl.cos(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # 4. Boundary conditions: u = 0 and w = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary(x_coord):
        return (np.isclose(x_coord[0], 0.0) | np.isclose(x_coord[0], 1.0) |
                np.isclose(x_coord[1], 0.0) | np.isclose(x_coord[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f, w = 0 on ∂Ω
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem1.solve()
    
    # Get iterations from step 1
    ksp1 = problem1.solver
    iter1 = ksp1.getIterationNumber()
    total_iterations += iter1
    
    # Step 2: Solve -Δu = w, u = 0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test2) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    iter2 = ksp2.getIterationNumber()
    total_iterations += iter2
    
    # 7. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
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
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }