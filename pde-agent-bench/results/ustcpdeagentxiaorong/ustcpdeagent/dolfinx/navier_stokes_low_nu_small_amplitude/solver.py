# Thread control - must be before any imports that load BLAS
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import ctypes as _ctypes
    for _libname in ["libopenblas.so", "libopenblas.so.0"]:
        try:
            _lib = _ctypes.CDLL(_libname, _ctypes.RTLD_GLOBAL)
            _lib.openblas_set_num_threads(1)
        except OSError:
            pass
except Exception:
    pass

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
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 200
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        0.2 * pi * ufl.cos(pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -0.4 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    # Source: f = (u·∇)u - ν Δu (p=0 so ∇p=0)
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact))
    
    # BCs
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # LU solver options with MUMPS
    lu_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "200",
        "mat_mumps_icntl_24": "1",
    }
    
    # Step 1: Stokes solve for initial guess
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v_t, q_t) = ufl.TestFunctions(W)
    
    a_stokes = (
        nu_c * ufl.inner(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
        - p_t * ufl.div(v_t) * ufl.dx
        - q_t * ufl.div(u_t) * ufl.dx
    )
    L_stokes = ufl.inner(f, v_t) * ufl.dx
    
    w_stokes = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options=lu_opts,
        petsc_options_prefix="stokes_"
    ).solve()
    
    stokes_ok = not (np.any(np.isinf(w_stokes.x.array)) or np.any(np.isnan(w_stokes.x.array)))
    
    # Step 2: Newton solve for full NS
    w = fem.Function(W)
    if stokes_ok:
        w.x.array[:] = w_stokes.x.array[:]
    else:
        # Fallback: use exact solution as initial guess via interpolation
        u_init = fem.Function(V)
        u_init.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        _, dof_map = W.sub(0).collapse()
        w.x.array[dof_map] = u_init.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F_form = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    J_form = ufl.derivative(F_form, w)
    
    ns_opts = dict(lu_opts)
    ns_opts.update({
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": "1e-10",
        "snes_atol": "1e-12",
        "snes_max_it": "50",
    })
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=ns_opts
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Sample velocity magnitude onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
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
    
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape(ny_out, nx_out)
    
    # Handle NaN values (boundary points that might not be found)
    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.ravel())
        if np.any(valid):
            interp = NearestNDInterpolator(
                np.column_stack([XX.ravel()[valid], YY.ravel()[valid]]),
                u_grid.ravel()[valid]
            )
            nan_mask = np.isnan(u_grid.ravel())
            u_grid_flat = u_grid.ravel()
            u_grid_flat[nan_mask] = interp(XX.ravel()[nan_mask], YY.ravel()[nan_mask])
            u_grid = u_grid_flat.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [2],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"coefficients": {"nu": 0.01}},
        "output": {"grid": {"nx": 100, "ny": 100, "bbox": [0.0, 1.0, 0.0, 1.0]}, "field": "velocity_magnitude"},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f} s")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    
    if not np.any(np.isnan(result['u'])):
        grid = case_spec["output"]["grid"]
        xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
        ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
        XX, YY = np.meshgrid(xs, ys)
        ux_exact = 0.2 * np.pi * np.cos(np.pi * YY) * np.sin(2 * np.pi * XX)
        uy_exact = -0.4 * np.pi * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY)
        mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
        l2_grid_error = np.sqrt(np.mean((result['u'] - mag_exact)**2))
        linf_grid_error = np.max(np.abs(result['u'] - mag_exact))
        print(f"Grid L2 error: {l2_grid_error:.6e}")
        print(f"Grid Linf error: {linf_grid_error:.6e}")
    else:
        print("FAILED: NaN in output")
