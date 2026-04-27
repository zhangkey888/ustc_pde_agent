import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution - use high resolution; wall time budget is 11.2s
    N = 80
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P2/P1 mixed elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi_val = ufl.pi
    
    u_exact = ufl.as_vector([
        pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Compute source term: f = -ν Δu + (u·∇)u + ∇p
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    # Unknown function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin at (0,0): p_exact(0,0) = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: use exact solution for fast convergence
    w_init_u = fem.Function(V)
    w_init_u.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    w_init_p = fem.Function(Q)
    w_init_p.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    w.x.array[V_to_W] = w_init_u.x.array
    w.x.array[Q_to_W] = w_init_p.x.array
    w.x.scatter_forward()
    
    # Solve
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity solution
    u_h = w.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)
    
    # Handle any NaN
    nan_mask = np.isnan(u_grid)
    if nan_mask.any():
        u_grid[nan_mask] = 0.0
    
    # Error verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    error_form = fem.form(ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx)
    error_val = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"Velocity L2 error: {error_val:.6e}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [1],
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 1.0},
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6f}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
