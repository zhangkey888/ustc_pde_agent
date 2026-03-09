import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes with manufactured solution."""
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    nu_val = case_spec.get("pde", {}).get("viscosity", 0.12)
    
    # Use high-order elements for accuracy: P4/P3 Taylor-Hood
    degree_u = 4
    degree_p = 3
    N = 44  # mesh resolution - gives ~5.6e-7 max error
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Create mixed element (Taylor-Hood)
    ve = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pe = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    me = basix.ufl.mixed_element([ve, pe])
    W = fem.functionspace(domain, me)
    
    # Create function and split
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    u_exact_0 = -40 * (x[1] - 0.5) * ufl.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    u_exact_1 = 40 * (x[0] - 0.5) * ufl.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Source term: f = (u_exact · ∇)u_exact - ν Δu_exact  (since p_exact = 0, ∇p = 0)
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact))
    
    # Residual: ν(∇u, ∇v) + ((u·∇)u, v) - (p, div(v)) + (div(u), q) - (f, v) = 0
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions
    V_sub, V_sub_map = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    u_bc_func.interpolate(lambda x: np.stack([
        -40 * (x[1] - 0.5) * np.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
        40 * (x[0] - 0.5) * np.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    ]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at a corner to fix the constant
    Q_sub, Q_sub_map = W.sub(1).collapse()
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.x.array[:] = 0.0
    
    def corner_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q_sub), corner_marker)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution for velocity (helps Newton converge in 0 iterations)
    w.sub(0).interpolate(lambda x: np.stack([
        -40 * (x[1] - 0.5) * np.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
        40 * (x[0] - 0.5) * np.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    ]))
    
    # Solve using NonlinearProblem (high-level SNES interface)
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_linesearch_type": "bt",
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w,
        petsc_options_prefix="ns_",
        bcs=bcs,
        petsc_options=petsc_opts,
    )
    
    problem.solve()
    
    snes = problem.solver
    newton_its = snes.getIterationNumber()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.flatten()
    points[1] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    output = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [newton_its],
        }
    }
    
    return output


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "viscosity": 0.12,
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    # Compute exact solution
    nx, ny = 50, 50
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux = -40*(YY-0.5)*np.exp(-20*((XX-0.5)**2 + (YY-0.5)**2))
    uy = 40*(XX-0.5)*np.exp(-20*((XX-0.5)**2 + (YY-0.5)**2))
    vel_mag_exact = np.sqrt(ux**2 + uy**2)
    
    u_grid = result['u']
    error = np.abs(u_grid - vel_mag_exact)
    max_error = np.nanmax(error)
    l2_error = np.sqrt(np.nanmean(error**2))
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Max error: {max_error:.2e}")
    print(f"L2 error: {l2_error:.2e}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")
