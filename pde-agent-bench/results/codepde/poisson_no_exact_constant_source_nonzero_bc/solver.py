import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract source term value
    source_term = pde_config.get("source_term", {})
    f_value = 1.0
    if isinstance(source_term, dict):
        f_value = float(source_term.get("value", 1.0))
    elif isinstance(source_term, (int, float)):
        f_value = float(source_term)
    
    # Extract diffusion coefficient
    coefficients = pde_config.get("coefficients", {})
    kappa_value = 1.0
    if isinstance(coefficients, dict):
        kappa_value = float(coefficients.get("kappa", coefficients.get("κ", 1.0)))
    
    # Extract boundary conditions
    bcs_spec = pde_config.get("boundary_conditions", {})
    
    # Grid size for output
    nx_out = 50
    ny_out = 50
    
    # 2. Create mesh
    mesh_res = 100
    element_deg = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_deg))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(f_value))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_value))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions
    # Parse BC - for "nonzero_bc" case, we need to figure out the BC function
    # Default: try to extract from case_spec
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Determine BC value/function
    # For "nonzero_bc", typical choice is g based on the spec
    # Check if there's a specific BC expression
    bc_value_str = None
    if isinstance(bcs_spec, dict):
        for key, val in bcs_spec.items():
            if isinstance(val, dict):
                bc_value_str = val.get("value", val.get("expression", None))
            elif isinstance(val, str):
                bc_value_str = val
    
    # Create BC function
    u_bc = fem.Function(V)
    
    if bc_value_str is not None and isinstance(bc_value_str, str):
        # Try to parse expression
        def make_bc_func(expr_str):
            def bc_func(x):
                x0 = x[0]
                x1 = x[1]
                local_dict = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "pi": np.pi,
                              "x[0]": x0, "x[1]": x1}
                try:
                    return eval(expr_str, {"__builtins__": {}}, local_dict)
                except:
                    return np.zeros(x.shape[1])
            return bc_func
        u_bc.interpolate(make_bc_func(bc_value_str))
    else:
        # Try numeric value
        if bc_value_str is not None:
            try:
                val = float(bc_value_str)
                u_bc.interpolate(lambda x: np.full(x.shape[1], val))
            except:
                u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
        else:
            # Default nonzero BC: use x*(1-x)*y*(1-y) won't work for boundary...
            # For "nonzero_bc" with constant source, a common choice is g = some function
            # Let's check more carefully
            g_value = 0.0
            if isinstance(bcs_spec, dict):
                for key, val in bcs_spec.items():
                    if isinstance(val, dict) and "value" in val:
                        try:
                            g_value = float(val["value"])
                        except (ValueError, TypeError):
                            pass
            
            # If still 0, check if "nonzero" is in the case name
            case_id = case_spec.get("case_id", "")
            if "nonzero_bc" in case_id and g_value == 0.0:
                # A typical nonzero BC: g = 1.0 or some expression
                # Let's try g = sin(pi*x)*sin(pi*y) on boundary... that's 0 on boundary
                # More likely: g = x + y, or g = 1, etc.
                # Check pde config more carefully
                bc_info = pde_config.get("bc", pde_config.get("boundary_conditions", {}))
                if isinstance(bc_info, dict):
                    for k, v in bc_info.items():
                        if isinstance(v, dict):
                            expr = v.get("expression", v.get("value", None))
                            if expr is not None:
                                if isinstance(expr, str):
                                    bc_value_str = expr
                                else:
                                    try:
                                        g_value = float(expr)
                                    except:
                                        pass
                
                if bc_value_str is not None and isinstance(bc_value_str, str):
                    def bc_func_parsed(x):
                        x0, x1 = x[0], x[1]
                        local_ns = {"x": x, "np": np, "sin": np.sin, "cos": np.cos,
                                    "pi": np.pi, "exp": np.exp}
                        expr_clean = bc_value_str.replace("x[0]", "x0").replace("x[1]", "x1")
                        local_ns["x0"] = x0
                        local_ns["x1"] = x1
                        try:
                            return eval(expr_clean, {"__builtins__": {}}, local_ns)
                        except:
                            return np.zeros(x.shape[1])
                    u_bc.interpolate(bc_func_parsed)
                else:
                    u_bc.interpolate(lambda x: np.full(x.shape[1], g_value))
            else:
                u_bc.interpolate(lambda x: np.full(x.shape[1], g_value))
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    x_grid = np.linspace(0.0, 1.0, nx_out)
    y_grid = np.linspace(0.0, 1.0, ny_out)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = xx.ravel()
    points[1, :] = yy.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }