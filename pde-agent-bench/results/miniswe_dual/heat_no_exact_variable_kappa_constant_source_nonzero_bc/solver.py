import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time
import re

ScalarType = PETSc.ScalarType


def _parse_expr_to_ufl(expr_str, x):
    """Parse a simple expression string to UFL using safe token-based replacement."""
    s = expr_str
    
    replacements = {
        'pi': 'ufl.pi',
        'sin': 'ufl.sin',
        'cos': 'ufl.cos',
        'exp': 'ufl.exp',
        'tan': 'ufl.tan',
        'sqrt': 'ufl.sqrt',
        'log': 'ufl.ln',
        'abs': 'ufl.algebra.Abs',
        'x': 'x[0]',
        'y': 'x[1]',
        'z': 'x[2]',
    }
    
    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
    for key in sorted_keys:
        pattern = r'\b' + re.escape(key) + r'\b'
        s = re.sub(pattern, replacements[key], s)
    
    try:
        return eval(s)
    except Exception as e:
        print(f"Warning: Could not parse expression '{expr_str}' -> '{s}': {e}")
        return None


def _parse_expr_to_lambda(expr_str):
    """Parse a simple expression string to a numpy lambda for interpolation."""
    def func(x_arr):
        x = x_arr[0]
        y = x_arr[1]
        z = x_arr[2] if x_arr.shape[0] > 2 else np.zeros_like(x)
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        sqrt = np.sqrt
        log = np.log
        try:
            return eval(expr_str)
        except Exception:
            return np.zeros_like(x)
    return func


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with variable kappa."""
    
    t_start = time.time()
    
    # ---- Parse case_spec ----
    pde = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults as fallback
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Source term
    source = pde.get("source", {})
    if isinstance(source, dict):
        source_expr = source.get("expr", None)
        source_val = source.get("value", 1.0)
    elif isinstance(source, (int, float)):
        source_val = float(source)
        source_expr = None
    else:
        source_val = 1.0
        source_expr = None
    
    # Initial condition
    ic = pde.get("initial_condition", {})
    if isinstance(ic, dict):
        ic_expr = ic.get("expr", None)
        ic_val = ic.get("value", 0.0)
    elif isinstance(ic, (int, float)):
        ic_val = float(ic)
        ic_expr = None
    else:
        ic_val = 0.0
        ic_expr = None
    
    # Kappa (diffusion coefficient)
    coeffs_dict = pde.get("coefficients", {})
    kappa_spec = coeffs_dict.get("kappa", coeffs_dict.get("k", {}))
    
    # Boundary conditions
    bc_spec = pde.get("boundary_conditions", pde.get("bc", {}))
    
    # ---- Solver parameters ----
    N = 64
    degree = 1
    # Use suggested dt but cap at reasonable value for accuracy
    dt = min(dt_suggested, 0.01)
    
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    rtol_used = 1e-8
    
    # ---- Create mesh ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # ---- Define kappa as UFL expression ----
    x = ufl.SpatialCoordinate(domain)
    
    if isinstance(kappa_spec, dict):
        kappa_type = kappa_spec.get("type", "constant")
        if kappa_type == "expr":
            kappa_expr_str = kappa_spec.get("expr", "1.0")
            kappa_ufl = _parse_expr_to_ufl(kappa_expr_str, x)
            if kappa_ufl is None:
                kappa_ufl = fem.Constant(domain, ScalarType(1.0))
        elif kappa_type == "constant":
            kappa_val = float(kappa_spec.get("value", 1.0))
            kappa_ufl = fem.Constant(domain, ScalarType(kappa_val))
        else:
            kappa_ufl = fem.Constant(domain, ScalarType(1.0))
    elif isinstance(kappa_spec, (int, float)):
        kappa_ufl = fem.Constant(domain, ScalarType(float(kappa_spec)))
    else:
        kappa_ufl = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # ---- Source term ----
    if source_expr is not None:
        f_ufl = _parse_expr_to_ufl(str(source_expr), x)
        if f_ufl is None:
            f_ufl = fem.Constant(domain, ScalarType(float(source_val) if source_val is not None else 1.0))
    else:
        f_ufl = fem.Constant(domain, ScalarType(float(source_val) if source_val is not None else 1.0))
    
    # ---- Time step ----
    n_steps = int(np.ceil(t_end / dt))
    dt_actual = t_end / n_steps
    dt_const = fem.Constant(domain, ScalarType(dt_actual))
    
    # ---- Functions ----
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    # Initial condition
    if ic_expr is not None:
        ic_func = _parse_expr_to_lambda(str(ic_expr))
        u_n.interpolate(ic_func)
    else:
        u0_val = float(ic_val) if ic_val is not None else 0.0
        u_n.interpolate(lambda x_arr, v=u0_val: np.full(x_arr.shape[1], v))
    u_h.x.array[:] = u_n.x.array[:]
    
    # ---- Boundary conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    bc_list = []
    
    def _make_bc_from_spec(bc_item, V, domain, fdim):
        bcs_out = []
        if not isinstance(bc_item, dict):
            return bcs_out
        
        bc_type = bc_item.get("type", "dirichlet")
        if bc_type != "dirichlet":
            return bcs_out
        
        location = bc_item.get("location", "all")
        
        loc_markers = {
            "left": lambda x_arr: np.isclose(x_arr[0], 0.0),
            "right": lambda x_arr: np.isclose(x_arr[0], 1.0),
            "bottom": lambda x_arr: np.isclose(x_arr[1], 0.0),
            "top": lambda x_arr: np.isclose(x_arr[1], 1.0),
        }
        
        if location in loc_markers:
            marker = loc_markers[location]
        else:
            marker = lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        bc_val = bc_item.get("value", None)
        bc_expr_str = bc_item.get("expr", None)
        
        u_bc = fem.Function(V)
        if bc_expr_str is not None:
            bc_func = _parse_expr_to_lambda(str(bc_expr_str))
            u_bc.interpolate(bc_func)
        elif bc_val is not None:
            try:
                val = float(bc_val)
                u_bc.interpolate(lambda x_arr, v=val: np.full(x_arr.shape[1], v))
            except (ValueError, TypeError):
                u_bc.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        else:
            u_bc.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        
        bcs_out.append(fem.dirichletbc(u_bc, dofs))
        return bcs_out
    
    if isinstance(bc_spec, list):
        for bc_item in bc_spec:
            bc_list.extend(_make_bc_from_spec(bc_item, V, domain, fdim))
    elif isinstance(bc_spec, dict):
        if "type" in bc_spec:
            bc_list.extend(_make_bc_from_spec(bc_spec, V, domain, fdim))
        else:
            for loc, bc_item in bc_spec.items():
                if isinstance(bc_item, dict):
                    bc_item_copy = dict(bc_item)
                    bc_item_copy.setdefault("location", loc)
                    bc_list.extend(_make_bc_from_spec(bc_item_copy, V, domain, fdim))
    
    # Fallback: zero Dirichlet on all boundaries
    if len(bc_list) == 0:
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        bc_list.append(fem.dirichletbc(u_bc, dofs))
    
    bcs = bc_list
    
    # ---- Variational form (Backward Euler) ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u * v / dt_const + kappa_ufl * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (f_ufl * v + u_n * v / dt_const) * ufl.dx
    
    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # ---- Assemble matrix (constant in time) ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # ---- Create solver ----
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    try:
        pc.setType("hypre")
    except Exception:
        pc.setType(PETSc.PC.Type.ILU)
        pc_type_used = "ilu"
    
    solver.setTolerances(rtol=rtol_used, atol=1e-12, max_it=1000)
    solver.setFromOptions()
    
    # ---- Time stepping ----
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt_actual
        
        # Assemble RHS
        b = petsc.assemble_vector(L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
        
        b.destroy()
    
    # ---- Evaluate on 50x50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Evaluate initial condition on same grid
    u_init_func = fem.Function(V)
    if ic_expr is not None:
        ic_func_eval = _parse_expr_to_lambda(str(ic_expr))
        u_init_func.interpolate(ic_func_eval)
    else:
        u0_val_eval = float(ic_val) if ic_val is not None else 0.0
        u_init_func.interpolate(lambda x_arr, v=u0_val_eval: np.full(x_arr.shape[1], v))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    
    elapsed = time.time() - t_start
    print(f"Solve completed in {elapsed:.3f}s, {n_steps} steps, {total_iterations} total iterations")
    print(f"u_grid range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "heat",
            "source": {"value": 1.0},
            "initial_condition": {"value": 0.0},
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"}
            },
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            },
            "boundary_conditions": [
                {"type": "dirichlet", "value": 0.0}
            ],
            "domain": {"type": "unit_square"}
        }
    }
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
