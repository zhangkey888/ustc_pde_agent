import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde_spec = case_spec.get("pde", {})
    
    # Time parameters with hardcoded defaults
    t_end = 0.15
    dt_val = 0.005
    time_scheme = "backward_euler"
    is_transient = True
    
    time_spec = pde_spec.get("time", None)
    if time_spec is not None:
        t_end = time_spec.get("t_end", t_end)
        dt_val = time_spec.get("dt", dt_val)
        time_scheme = time_spec.get("scheme", time_scheme)
    
    epsilon = pde_spec.get("epsilon", 1.0)
    
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 70)
    ny_out = output_spec.get("ny", 70)
    
    # Mesh and FE parameters
    N = 64
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * 0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = du/dt - eps*laplacian(u) + u^3 - u
    du_dt_exact = -ufl.exp(-t_const) * 0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    laplacian_u_exact = ufl.exp(-t_const) * 0.3 * (-2.0 * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    R_u_exact = u_exact_ufl**3 - u_exact_ufl
    f_source = du_dt_exact - epsilon * laplacian_u_exact + R_u_exact
    
    # Functions
    u_h = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    
    # Initial condition
    u_init_expr = fem.Expression(
        0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)
    u_h.interpolate(u_init_expr)
    
    # Boundary conditions (u=0 on boundary of unit square for this manufactured solution)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]
    
    # Backward Euler weak form
    F_form = (
        (u_h - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + (u_h**3 - u_h) * v * ufl.dx
        - f_source * v * ufl.dx
    )
    
    # Time stepping
    n_steps = int(round(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_c.value = actual_dt
    
    t = 0.0
    nonlinear_iterations = []
    
    # Create NonlinearProblem with new API
    problem = petsc.NonlinearProblem(
        F_form, u_h,
        petsc_options_prefix="nls_",
        bcs=bcs,
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "snes_error_if_not_converged": False,
        },
    )
    
    for step in range(n_steps):
        t += actual_dt
        t_const.value = t
        
        problem.solve()
        
        snes = problem.solver
        reason = snes.getConvergedReason()
        n_iters = snes.getIterationNumber()
        assert reason > 0, f"SNES did not converge at step {step}, reason={reason}"
        
        u_h.x.scatter_forward()
        nonlinear_iterations.append(n_iters)
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on output grid
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Initial condition on output grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_init_expr)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": 0,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iterations,
        },
        "u_initial": u_initial_grid,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "epsilon": 1.0,
            "time": {
                "t_end": 0.15,
                "dt": 0.005,
                "scheme": "backward_euler",
            },
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 70, "ny": 70},
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    u_grid = result["u"]
    print(f"Shape: {u_grid.shape}")
    print(f"Range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Time: {elapsed:.2f}s")
    
    xs = np.linspace(0.0, 1.0, 70)
    ys = np.linspace(0.0, 1.0, 70)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.15) * 0.3 * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    diff = u_grid - u_exact
    valid = ~np.isnan(diff)
    l2_err = np.sqrt(np.mean(diff[valid]**2))
    linf_err = np.max(np.abs(diff[valid]))
    print(f"L2 error: {l2_err:.6e}")
    print(f"Linf error: {linf_err:.6e}")
    print(f"Newton iters: {result['solver_info']['nonlinear_iterations']}")
