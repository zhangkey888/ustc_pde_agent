import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 0.25
    N = 16
    degree_u = 3
    degree_p = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Create elements
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mel)
    
    # Also create individual spaces for BCs
    V = fem.functionspace(domain, ("Lagrange", degree_u, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Define solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution (for source term computation)
    u_exact = ufl.as_vector([
        x[0]*(1 - x[0])*(1 - 2*x[1]),
        -x[1]*(1 - x[1])*(1 - 2*x[0])
    ])
    p_exact = x[0] - x[1]
    
    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    f = (ufl.grad(u_exact) * u_exact 
         - nu * ufl.div(ufl.grad(u_exact)) 
         + ufl.grad(p_exact))
    
    # Weak form (residual)
    F_form = (
        ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        x[0]*(1 - x[0])*(1 - 2*x[1]),
        -x[1]*(1 - x[1])*(1 - 2*x[0])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure pin at origin to remove nullspace
    def origin(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda x: x[0] - x[1])
    
    origin_vertices = mesh.locate_entities_boundary(domain, 0, origin)
    dofs_p = fem.locate_dofs_topological((W.sub(1), Q), 0, origin_vertices)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.stack([
        x[0]*(1 - x[0])*(1 - 2*x[1]),
        -x[1]*(1 - x[1])*(1 - 2*x[0])
    ]))
    p_init = fem.Function(Q)
    p_init.interpolate(lambda x: x[0] - x[1])
    
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    problem.solve()
    
    snes = problem.solver
    newton_its = snes.getIterationNumber()
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.zeros(nx_eval * ny_eval)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[i] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [newton_its],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Newton iterations: {result['solver_info']['nonlinear_iterations']}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Min/Max velocity magnitude: {u_grid.min():.6e}, {u_grid.max():.6e}")
    
    # Compute exact velocity magnitude on same grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = XX*(1-XX)*(1-2*YY)
    uy_exact = -YY*(1-YY)*(1-2*XX)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    error = np.max(np.abs(u_grid - vel_mag_exact))
    print(f"Max error vs exact: {error:.6e}")
    
    l2_error = np.sqrt(np.mean((u_grid - vel_mag_exact)**2))
    print(f"L2 error vs exact: {l2_error:.6e}")
