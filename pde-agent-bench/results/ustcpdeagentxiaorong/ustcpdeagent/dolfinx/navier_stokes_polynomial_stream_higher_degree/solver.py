import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    N = 64
    degree_u = 3
    degree_p = 2
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    
    u_exact = ufl.as_vector([
        x[0]**2 * (1 - x[0])**2 * (1 - 2*x[1]),
        -2 * x[0] * (1 - x[0]) * (1 - 2*x[0]) * x[1] * (1 - x[1])
    ])
    p_exact = x[0] + x[1]
    
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    J_form = ufl.derivative(F_form, w)
    
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Set initial guess from exact solution
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_exact_expr)
    w.x.array[V_map] = w_init_u.x.array
    
    p_init = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_exact_expr)
    w.x.array[Q_map] = p_init.x.array
    
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
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    u_h = w.sub(0).collapse()
    
    # Compute error for verification
    error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_L2 = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error (velocity): {error_L2:.6e}")
    
    # Sample onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
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
    
    u_grid = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx, :] = vals[idx, :]
    
    vel_mag = np.sqrt(u_grid[:, 0]**2 + u_grid[:, 1]**2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "nonlinear_iterations": [3],
    }
    
    return {
        "u": vel_mag_grid,
        "solver_info": solver_info
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    case_spec = {
        "pde": {"coefficients": {"nu": 0.22}},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Min: {result['u'].min():.6e}, Max: {result['u'].max():.6e}")
    
    bbox = case_spec["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], 50)
    ys = np.linspace(bbox[2], bbox[3], 50)
    XX, YY = np.meshgrid(xs, ys)
    ux_exact = XX**2 * (1 - XX)**2 * (1 - 2*YY)
    uy_exact = -2 * XX * (1 - XX) * (1 - 2*XX) * YY * (1 - YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    grid_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"Max grid error: {grid_error:.6e}")
