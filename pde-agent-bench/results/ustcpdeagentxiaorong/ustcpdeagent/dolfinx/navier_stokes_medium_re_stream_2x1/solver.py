import numpy as np
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
    
    N = 96
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi_sym = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        pi_sym * ufl.cos(pi_sym * x[1]) * ufl.sin(2 * pi_sym * x[0]),
        -2 * pi_sym * ufl.cos(2 * pi_sym * x[0]) * ufl.sin(pi_sym * x[1])
    ])
    p_exact = ufl.sin(pi_sym * x[0]) * ufl.cos(pi_sym * x[1])
    
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Source term: f = -nu * laplacian(u_exact) + (u_exact . grad) u_exact + grad(p_exact)
    f_expr = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    # Nonlinear residual
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    
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
    
    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Step 1: Stokes solve as initial guess
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    
    F_stokes = (
        nu_c * ufl.inner(ufl.grad(u_s), ufl.grad(v)) * ufl.dx
        - p_s * ufl.div(v) * ufl.dx
        + ufl.div(u_s) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J_stokes = ufl.derivative(F_stokes, w_stokes)
    
    stokes_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    stokes_problem = petsc.NonlinearProblem(
        F_stokes, w_stokes, bcs=bcs, J=J_stokes,
        petsc_options_prefix="stokes_",
        petsc_options=stokes_opts
    )
    stokes_problem.solve()
    w_stokes.x.scatter_forward()
    
    # Use Stokes solution as initial guess for NS
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    # Clean up Stokes problem
    del stokes_problem
    
    # Step 2: Full NS Newton solve
    ns_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    ns_problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=ns_opts
    )
    ns_problem.solve()
    w.x.scatter_forward()
    
    reason_ns = ns_problem.solver.getConvergedReason()
    its_ns = ns_problem.solver.getIterationNumber()
    
    # Extract solution
    u_h = w.sub(0).collapse()
    
    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_2d = np.column_stack([XX.ravel(), YY.ravel()])
    pts_3d = np.zeros((pts_2d.shape[0], 3))
    pts_3d[:, :2] = pts_2d
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full((len(pts_3d), gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_eval, cells_eval)
        u_grid[eval_map] = vals
    
    vel_mag = np.sqrt(u_grid[:, 0]**2 + u_grid[:, 1]**2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [its_ns],
    }
    
    return {
        "u": vel_mag_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.2},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Min vel mag: {np.nanmin(result['u']):.6e}")
    print(f"Max vel mag: {np.nanmax(result['u']):.6e}")
    
    # Verify against exact solution
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(2 * np.pi * XX)
    uy_exact = -2 * np.pi * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    err = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    rms_err = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    print(f"Max pointwise error on grid: {err:.6e}")
    print(f"RMS error on grid: {rms_err:.6e}")
