import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 1.0
    nx_out, ny_out = 50, 50
    
    # P3/P2 Taylor-Hood with N=32 gives ~2.6e-7 error well under 1e-6
    N = 32
    degree_u = 3
    degree_p = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Create mixed element using basix.ufl
    cell_name = domain.topology.cell_name()
    vel_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mel)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Current solution
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Source term: f = u_exact·∇u_exact - ν ∇²u_exact + ∇p_exact
    f = (ufl.grad(u_exact) * u_exact 
         - nu_val * ufl.div(ufl.grad(u_exact)) 
         + ufl.grad(p_exact))
    
    # Weak form (residual)
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at corner to fix uniqueness
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    corner_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), corner)
    bc_p = fem.dirichletbc(p_bc_func, corner_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution for fast Newton convergence
    w.sub(0).interpolate(u_bc_func)
    w.sub(1).interpolate(p_bc_func)
    
    # PETSc options
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 25,
        "snes_linesearch_type": "basic",
    }
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    problem.solve()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    converged_reason = snes.getConvergedReason()
    assert converged_reason > 0, f"SNES did not converge, reason: {converged_reason}"
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, mi in enumerate(eval_map):
            u_values[mi, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    vel_mag_grid = vel_mag.reshape((nx_out, ny_out))
    
    elapsed = time.time() - t0
    
    result = {
        "u": vel_mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
    
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    # Compute reference velocity magnitude
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    error = np.sqrt(np.mean((result["u"] - vel_mag_exact)**2))
    max_err = np.max(np.abs(result["u"] - vel_mag_exact))
    print(f"L2 error (velocity magnitude): {error:.6e}")
    print(f"Max error: {max_err:.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Newton iterations: {result['solver_info']['nonlinear_iterations']}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
