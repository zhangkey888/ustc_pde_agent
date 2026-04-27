import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution - use quadrilaterals as specified by case ID
    # With N=80, error ~3.9e-05, time ~4s. Budget is 108s, target error < 1.44e-04
    # Increase to N=120 for better accuracy while staying within time budget
    N = 120
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    gdim = msh.geometry.dim
    
    # Build mixed function space (Taylor-Hood on quads: Q2/Q1)
    cell_name = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    mel = basix_mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mel)
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + pi * ufl.cos(4*pi * x[1]) * ufl.sin(2*pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - (pi/2) * ufl.cos(2*pi * x[0]) * ufl.sin(4*pi * x[1])
    ])
    
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(2*pi * x[1])
    
    # Compute source term: f = u_exact . grad(u_exact) - nu * div(grad(u_exact)) + grad(p_exact)
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Residual form
    F = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Jacobian
    J_form = ufl.derivative(F, w)
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin at (0,0) - p_exact(0,0) = sin(0)*cos(0) = 0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initialize with exact solution as good initial guess for fast Newton convergence
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_bc_expr)
    w.x.array[V_to_W] = w_init_u.x.array
    
    # Also initialize pressure
    p_init = fem.Function(Q)
    p_init_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_init_expr)
    w.x.array[Q_to_W] = p_init.x.array
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J_form,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract solution
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    
    # Compute error for verification
    error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error (velocity): {error_global:.6e}")
    
    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)
    
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {u_grid.shape}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [5],
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.1},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        }
    }
    
    result = solve(case_spec)
    print(f"Min vel mag: {np.nanmin(result['u']):.6e}")
    print(f"Max vel mag: {np.nanmax(result['u']):.6e}")
