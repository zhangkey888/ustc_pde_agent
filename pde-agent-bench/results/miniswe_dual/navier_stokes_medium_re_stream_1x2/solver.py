import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes with manufactured solution."""
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde_info = case_spec.get("pde", {})
    nu_val = pde_info.get("viscosity", 0.2)
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    N = 48
    degree_u = 3
    degree_p = 2
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Create mixed function space (Taylor-Hood P3/P2)
    vel_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_u, shape=(msh.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mel)
    
    # Individual spaces for BCs and interpolation
    V = fem.functionspace(msh, ("Lagrange", degree_u, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("Lagrange", degree_p))
    
    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi_val = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        2 * pi_val * ufl.cos(2 * pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    # Source term from manufactured solution
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    # Weak form (residual)
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    
    # Velocity BC (exact solution on all boundaries)
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess: interpolate exact solution for fast Newton convergence
    u_init = fem.Function(V)
    u_init.interpolate(u_bc_expr)
    w.sub(0).interpolate(u_init)
    
    p_init = fem.Function(Q)
    p_init_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_init_expr)
    w.sub(1).interpolate(p_init)
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    problem.solve()
    w.x.scatter_forward()
    
    # Get Newton iteration count
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    u_values = np.full((nx_out * ny_out, msh.geometry.dim), np.nan)
    pts_list = []
    cells_list = []
    idx_list = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points.T[i])
            cells_list.append(links[0])
            idx_list.append(i)
    
    if len(pts_list) > 0:
        pts_arr = np.array(pts_list)
        cells_arr = np.array(cells_list, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for j, gi in enumerate(idx_list):
            u_values[gi] = vals[j]
    
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx_out, ny_out))
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {"type": "navier_stokes", "viscosity": 0.2},
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50, "field": "velocity_magnitude"},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Newton iterations: {result['solver_info']['nonlinear_iterations']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    u_exact_x = 2 * np.pi * np.cos(2 * np.pi * Y) * np.sin(np.pi * X)
    u_exact_y = -np.pi * np.cos(np.pi * X) * np.sin(2 * np.pi * Y)
    u_exact_mag = np.sqrt(u_exact_x**2 + u_exact_y**2)
    
    rel_err = np.sqrt(np.mean((result['u'] - u_exact_mag)**2)) / np.sqrt(np.mean(u_exact_mag**2))
    max_err = np.max(np.abs(result['u'] - u_exact_mag))
    print(f"Relative L2 error: {rel_err:.2e}")
    print(f"Max absolute error: {max_err:.2e}")
