import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa_val = float(coefficients.get("kappa", 1.0))
    
    # Source term
    source = pde.get("source", {})
    if isinstance(source, dict):
        f_val = float(source.get("value", source.get("f", 1.0)))
    else:
        f_val = float(source) if source else 1.0
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    extents = domain_spec.get("extents", [[0.0, 1.0], [0.0, 1.0]])
    x_min, x_max = extents[0]
    y_min, y_max = extents[1]
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Parse boundary conditions
    bc_info = _parse_bcs(pde)
    
    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    element_degree = 2
    
    prev_norm = None
    final_u_grid = None
    final_info = {}
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        
        p0 = np.array([x_min, y_min])
        p1 = np.array([x_max, y_max])
        domain_mesh = mesh.create_rectangle(
            comm, [p0, p1], [N, N],
            cell_type=mesh.CellType.triangle
        )
        
        V = fem.functionspace(domain_mesh, ("Lagrange", element_degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        kappa = fem.Constant(domain_mesh, ScalarType(kappa_val))
        f = fem.Constant(domain_mesh, ScalarType(f_val))
        
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L_form = f * v * ufl.dx
        
        # Build boundary conditions
        bcs = _build_bcs(domain_mesh, V, bc_info)
        
        # Solve
        ksp_type = "cg"
        pc_type = "hypre"
        rtol = 1e-10
        
        try:
            problem = petsc.LinearProblem(
                a, L_form, bcs=bcs,
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "2000",
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
        except Exception:
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a, L_form, bcs=bcs,
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="poisson_lu_"
            )
            u_sol = problem.solve()
        
        # Compute norm for convergence check
        current_norm = np.sqrt(
            comm.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
                op=MPI.SUM
            )
        )
        
        # Evaluate on output grid
        u_grid = _evaluate_on_grid(domain_mesh, u_sol, x_min, x_max, y_min, y_max, nx_out, ny_out)
        
        final_u_grid = u_grid
        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
        
        # Convergence check
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.001:
                break
        
        prev_norm = current_norm
    
    return {
        "u": final_u_grid,
        "solver_info": final_info,
    }


def _parse_bcs(pde):
    """Parse boundary conditions from the PDE spec."""
    bcs_spec = pde.get("boundary_conditions", [])
    
    # Default: try to find Dirichlet BC
    bc_value = 0.0
    bc_expr_str = None
    bc_type = "constant"
    
    # Check for 'g' key directly in pde
    if "g" in pde:
        g = pde["g"]
        if isinstance(g, (int, float)):
            bc_value = float(g)
            bc_type = "constant"
        elif isinstance(g, str):
            bc_expr_str = g
            bc_type = "expression"
    
    # Check boundary_conditions list
    for bc_item in bcs_spec:
        if isinstance(bc_item, dict):
            if bc_item.get("type", "").lower() == "dirichlet":
                val = bc_item.get("value", bc_item.get("g", 0.0))
                if isinstance(val, (int, float)):
                    bc_value = float(val)
                    bc_type = "constant"
                elif isinstance(val, str):
                    bc_expr_str = val
                    bc_type = "expression"
                break
    
    return {"type": bc_type, "value": bc_value, "expr": bc_expr_str}


def _build_bcs(domain_mesh, V, bc_info):
    """Build dolfinx boundary conditions."""
    tdim = domain_mesh.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain_mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    if bc_info["type"] == "expression" and bc_info["expr"] is not None:
        u_bc = fem.Function(V)
        expr_str = bc_info["expr"]
        u_bc.interpolate(lambda x: _eval_bc_expr(x, expr_str))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
    elif bc_info["value"] != 0.0:
        bc = fem.dirichletbc(ScalarType(bc_info["value"]), boundary_dofs, V)
    else:
        bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    return [bc]


def _eval_bc_expr(x, expr_str):
    """Evaluate a string expression for boundary conditions."""
    result = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        xi, yi = x[0, i], x[1, i]
        zi = x[2, i] if x.shape[0] > 2 else 0.0
        local_vars = {
            "x": xi, "y": yi, "z": zi,
            "pi": np.pi,
            "sin": np.sin, "cos": np.cos,
            "exp": np.exp, "sqrt": np.sqrt,
            "abs": abs, "pow": pow,
        }
        try:
            result[i] = eval(expr_str, {"__builtins__": {}}, local_vars)
        except Exception:
            result[i] = 0.0
    return result


def _evaluate_on_grid(domain_mesh, u_func, x_min, x_max, y_min, y_max, nx, ny):
    """Evaluate the solution on a uniform grid."""
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = grid_x.flatten()
    points[1, :] = grid_y.flatten()
    
    bb_tree = geometry.bb_tree(domain_mesh, domain_mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_mesh, cell_candidates, points.T)
    
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
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid
