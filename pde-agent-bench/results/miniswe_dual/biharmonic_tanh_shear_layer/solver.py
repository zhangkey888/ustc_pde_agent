import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """
    Solve the biharmonic equation using two sequential Poisson solves:
      Delta^2 u = f  in Omega,  u = g on dOmega
    
    Manufactured solution: u = tanh(6*(y-0.5))*sin(pi*x)
    
    Mixed formulation:
      1) -Delta sigma = f,  sigma = -Delta u_exact on dOmega
      2) -Delta u = sigma,  u = u_exact on dOmega
    """
    comm = MPI.COMM_WORLD
    
    N = 48
    deg = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = tanh(6*(y-0.5))*sin(pi*x)
    u_exact_ufl = ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0])
    
    # Source term f = Delta^2 u (computed symbolically via UFL)
    laplacian_u = ufl.div(ufl.grad(u_exact_ufl))
    f_expr = ufl.div(ufl.grad(laplacian_u))
    
    # sigma_exact = -Delta u for boundary condition on sigma
    sigma_exact_ufl = -laplacian_u
    
    # Boundary setup
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    total_iterations = 0
    
    # --- Solve 1: -Delta sigma = f, sigma|dOmega = sigma_exact ---
    sigma_trial = ufl.TrialFunction(V)
    tau = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(sigma_trial), ufl.grad(tau)) * ufl.dx
    L1 = ufl.inner(f_expr, tau) * ufl.dx
    
    sigma_bc_func = fem.Function(V)
    sigma_bc_expr = fem.Expression(sigma_exact_ufl, V.element.interpolation_points)
    sigma_bc_func.interpolate(sigma_bc_expr)
    
    bc_sigma = fem.dirichletbc(sigma_bc_func, boundary_dofs)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_sigma],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"},
        petsc_options_prefix="poisson1_"
    )
    sigma_h = problem1.solve()
    total_iterations += problem1.solver.getIterationNumber()
    
    # --- Solve 2: -Delta u = sigma_h, u|dOmega = u_exact ---
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(sigma_h, v_test) * ufl.dx
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"},
        petsc_options_prefix="poisson2_"
    )
    u_h = problem2.solve()
    total_iterations += problem2.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
        }
    }


if __name__ == "__main__":
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact_grid = np.tanh(6*(YY - 0.5)) * np.sin(np.pi * XX)
    
    diff = result['u'] - u_exact_grid
    max_err = np.nanmax(np.abs(diff))
    rms_err = np.sqrt(np.nanmean(diff**2))
    print(f"Mesh: {result['solver_info']['mesh_resolution']}, Degree: {result['solver_info']['element_degree']}")
    print(f"Max error on grid: {max_err:.6e}")
    print(f"RMS error on grid: {rms_err:.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Iterations: {result['solver_info']['iterations']}")
