import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    start_time = time_module.time()
    
    pde = case_spec.get("pde", {})
    source_expr_str = pde.get("source_term", "0")
    reaction_expr_str = pde.get("reaction", None)
    epsilon = float(pde.get("epsilon", pde.get("diffusion_coefficient", 1.0)))
    
    domain_spec = pde.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 70)
    ny_out = output_spec.get("ny", 70)
    
    # Time parameters with hardcoded defaults for this problem
    time_spec = pde.get("time", None)
    t_end = 0.3
    dt = 0.005
    scheme = "backward_euler"
    is_transient = True
    
    if time_spec is not None:
        t_end = float(time_spec.get("t_end", t_end))
        dt = float(time_spec.get("dt", dt))
        scheme = time_spec.get("scheme", scheme)
    
    ic_str = pde.get("initial_condition", "0")
    bc_spec = pde.get("boundary_conditions", {})
    bc_value_str = bc_spec.get("value", "0")
    
    # Use a good resolution - P2 elements on 64x64 mesh
    N = 64
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    
    # Parse source term
    f_expr = _parse_ufl_expr(source_expr_str, x, domain)
    
    # Functions
    u = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    
    # Initial condition
    if ic_str is not None and ic_str.strip() != "0" and ic_str.strip() != "":
        ic_ufl = _parse_ufl_expr(ic_str, x, domain)
        ic_fem = fem.Expression(ic_ufl, V.element.interpolation_points)
        u_n.interpolate(ic_fem)
        u.x.array[:] = u_n.x.array[:]
    
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    # BCs - Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xc: np.ones(xc.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_val = 0.0
    try:
        bc_val = float(bc_value_str)
    except (ValueError, TypeError):
        bc_val = 0.0
    
    bc = fem.dirichletbc(ScalarType(bc_val), dofs, V)
    bcs = [bc]
    
    # Reaction term
    R_u = _parse_reaction_ufl(reaction_expr_str, u, x, domain)
    
    # Variational form - backward Euler
    if is_transient:
        F = (
            (u - u_n) / dt_c * v * ufl.dx
            + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + R_u * v * ufl.dx
            - f_expr * v * ufl.dx
        )
    else:
        F = (
            eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + R_u * v * ufl.dx
            - f_expr * v * ufl.dx
        )
    
    # PETSc options
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
    }
    
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol_val = 1e-8
    
    nonlinear_iters = []
    
    if is_transient:
        n_steps = int(round(t_end / dt))
        actual_dt = t_end / n_steps
        dt_c.value = actual_dt
        
        problem = petsc.NonlinearProblem(
            F, u, bcs=bcs,
            petsc_options_prefix="nls_",
            petsc_options=petsc_options,
        )
        
        t = 0.0
        for step in range(n_steps):
            t += actual_dt
            problem.solve()
            snes = problem.solver
            n_newton = snes.getIterationNumber()
            u.x.scatter_forward()
            nonlinear_iters.append(int(n_newton))
            u_n.x.array[:] = u.x.array[:]
    else:
        n_steps = 0
        actual_dt = dt
        problem = petsc.NonlinearProblem(
            F, u, bcs=bcs,
            petsc_options_prefix="nls_",
            petsc_options=petsc_options,
        )
        problem.solve()
        snes = problem.solver
        n_newton = snes.getIterationNumber()
        u.x.scatter_forward()
        nonlinear_iters.append(int(n_newton))
    
    # Evaluate on output grid
    u_grid = _eval_on_grid(domain, u, x_range, y_range, nx_out, ny_out)
    u_init_grid = _eval_on_grid(domain, u_initial, x_range, y_range, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol_val,
        "nonlinear_iterations": nonlinear_iters,
    }
    
    if is_transient:
        solver_info["dt"] = actual_dt
        solver_info["n_steps"] = n_steps
        solver_info["time_scheme"] = scheme
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_init_grid,
    }
    
    return result


def _parse_ufl_expr(expr_str, x, domain):
    if expr_str is None or str(expr_str).strip() == "0" or str(expr_str).strip() == "":
        return fem.Constant(domain, ScalarType(0.0))
    
    safe = str(expr_str).replace("^", "**")
    
    ns = {
        "x": x[0], "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin, "cos": ufl.cos,
        "exp": ufl.exp, "sqrt": ufl.sqrt,
        "abs": ufl.algebra.Abs,
    }
    
    try:
        return eval(safe, {"__builtins__": {}}, ns)
    except Exception:
        return fem.Constant(domain, ScalarType(0.0))


def _parse_reaction_ufl(reaction_str, u_func, x, domain):
    if reaction_str is None or str(reaction_str).strip() == "0" or str(reaction_str).strip() == "":
        return fem.Constant(domain, ScalarType(0.0))
    
    safe = str(reaction_str).replace("^", "**")
    
    ns = {
        "u": u_func,
        "x": x[0], "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin, "cos": ufl.cos,
        "exp": ufl.exp, "sqrt": ufl.sqrt,
    }
    
    try:
        return eval(safe, {"__builtins__": {}}, ns)
    except Exception:
        return fem.Constant(domain, ScalarType(0.0))


def _eval_on_grid(domain, u_func, x_range, y_range, nx, ny):
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xx.flatten()
    points[1, :] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx * ny, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_values = np.nan_to_num(u_values, nan=0.0)
    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "source_term": "sin(6*pi*x)*sin(5*pi*y)",
            "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
            "epsilon": 0.01,
            "reaction": "u**3",
            "domain": {
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
            },
            "boundary_conditions": {
                "type": "dirichlet",
                "value": "0",
            },
            "time": {
                "t_end": 0.3,
                "dt": 0.005,
                "scheme": "backward_euler",
            },
        },
        "output": {
            "nx": 70,
            "ny": 70,
        },
    }
    
    t0 = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - t0
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"u norm: {np.linalg.norm(result['u']):.6f}")
    print(f"solver_info: {result['solver_info']}")
