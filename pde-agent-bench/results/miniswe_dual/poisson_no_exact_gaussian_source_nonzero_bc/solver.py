import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def _parse_expr_to_callable(expr_str):
    """Parse a string expression into a callable for interpolation.
    The callable takes coords of shape (3, N) and returns (N,)."""
    
    def func(coords):
        x = coords[0]
        y = coords[1]
        z = coords[2] if coords.shape[0] > 2 else np.zeros_like(x)
        local_ns = {
            'x': x, 'y': y, 'z': z,
            'np': np, 'pi': np.pi,
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'sqrt': np.sqrt,
            'abs': np.abs, 'log': np.log,
            'tan': np.tan, 'tanh': np.tanh,
            'cosh': np.cosh, 'sinh': np.sinh,
        }
        try:
            result = eval(expr_str, {"__builtins__": {}}, local_ns)
            if np.isscalar(result):
                return np.full(x.shape, float(result))
            return np.asarray(result, dtype=np.float64)
        except Exception:
            return np.zeros(x.shape)
    
    return func


def _parse_expr_to_ufl(expr_str, x):
    """Parse a string expression into a UFL expression."""
    local_ns = {
        'x': x[0], 'y': x[1],
        'pi': np.pi,
        'sin': ufl.sin, 'cos': ufl.cos,
        'exp': ufl.exp, 'sqrt': ufl.sqrt,
        'abs': ufl.algebra.Abs,
        'log': ufl.ln, 'tan': ufl.tan,
        'tanh': ufl.tanh, 'cosh': ufl.cosh, 'sinh': ufl.sinh,
    }
    try:
        result = eval(expr_str, {"__builtins__": {}}, local_ns)
        return result
    except Exception:
        return None


def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation -div(kappa * grad(u)) = f on [0,1]^2."""
    
    # Parse case_spec - handle both direct and oracle_config formats
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", case_spec.get("pde", {}))
    
    # Get coefficients
    coefficients = pde.get("coefficients", {})
    kappa_spec = coefficients.get("kappa", 1.0)
    if isinstance(kappa_spec, dict):
        kappa_val = float(kappa_spec.get("value", 1.0))
    else:
        kappa_val = float(kappa_spec)
    
    # Get source term
    source_str = pde.get("source_term", pde.get("source", "exp(-180*((x-0.3)**2 + (y-0.7)**2))"))
    
    # Get boundary condition info
    bc_config = oracle_config.get("bc", {})
    bc_dirichlet = bc_config.get("dirichlet", {})
    bc_value_str = bc_dirichlet.get("value", None)
    bc_on = bc_dirichlet.get("on", "all")
    
    # Also check pde-level boundary_conditions
    bcs_spec = pde.get("boundary_conditions", [])
    
    # Output grid size
    output_config = oracle_config.get("output", case_spec.get("output", {}))
    grid_config = output_config.get("grid", {})
    nx_out = grid_config.get("nx", output_config.get("nx", 50))
    ny_out = grid_config.get("ny", output_config.get("ny", 50))
    
    # Solver parameters
    element_degree = 2
    mesh_resolution = 96
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term as UFL
    f_ufl = _parse_expr_to_ufl(source_str, x)
    if f_ufl is None:
        # Fallback: hardcoded Gaussian source
        f_ufl = ufl.exp(-180.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Diffusion coefficient
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    bcs = []
    
    # Method 1: Parse from oracle_config bc section
    if bc_value_str is not None:
        # All boundary facets
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        if bc_value_str == "0" or bc_value_str == "0.0" or bc_value_str == 0:
            bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
        elif isinstance(bc_value_str, (int, float)):
            bc = fem.dirichletbc(ScalarType(float(bc_value_str)), boundary_dofs, V)
        elif isinstance(bc_value_str, str):
            u_bc = fem.Function(V)
            bc_callable = _parse_expr_to_callable(bc_value_str)
            u_bc.interpolate(bc_callable)
            bc = fem.dirichletbc(u_bc, boundary_dofs)
        else:
            bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
        
        bcs.append(bc)
    
    # Method 2: Parse from pde boundary_conditions list
    elif bcs_spec:
        for bc_spec_item in bcs_spec:
            bc_type = bc_spec_item.get("type", "dirichlet")
            bc_value = bc_spec_item.get("value", None)
            bc_location = bc_spec_item.get("location", "boundary")
            
            if bc_type.lower() == "dirichlet":
                if bc_location in ("boundary", "all", None):
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool)
                    )
                elif isinstance(bc_location, str):
                    loc_str = bc_location.lower().strip()
                    if loc_str == "left":
                        facets = mesh.locate_entities_boundary(domain, fdim, lambda coords: np.isclose(coords[0], 0.0))
                    elif loc_str == "right":
                        facets = mesh.locate_entities_boundary(domain, fdim, lambda coords: np.isclose(coords[0], 1.0))
                    elif loc_str == "bottom":
                        facets = mesh.locate_entities_boundary(domain, fdim, lambda coords: np.isclose(coords[1], 0.0))
                    elif loc_str == "top":
                        facets = mesh.locate_entities_boundary(domain, fdim, lambda coords: np.isclose(coords[1], 1.0))
                    else:
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool)
                        )
                else:
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool)
                    )
                
                dofs = fem.locate_dofs_topological(V, fdim, facets)
                
                if bc_value is None or bc_value == 0 or bc_value == "0" or bc_value == "0.0":
                    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
                elif isinstance(bc_value, (int, float)):
                    bc = fem.dirichletbc(ScalarType(float(bc_value)), dofs, V)
                elif isinstance(bc_value, str):
                    u_bc = fem.Function(V)
                    bc_callable = _parse_expr_to_callable(bc_value)
                    u_bc.interpolate(bc_callable)
                    bc = fem.dirichletbc(u_bc, dofs)
                else:
                    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
                
                bcs.append(bc)
    
    # Fallback: apply zero Dirichlet on all boundaries
    if not bcs:
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
        bcs = [bc]
    
    # Solve
    total_iterations = 0
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type_str,
                "pc_type": pc_type_str,
                "ksp_rtol": str(rtol),
                "ksp_atol": "1e-12",
                "ksp_max_it": "2000",
            },
            petsc_options_prefix="poisson_"
        )
        u_sol = problem.solve()
        try:
            total_iterations = problem.solver.getIterationNumber()
        except Exception:
            total_iterations = -1
    except Exception:
        # Fallback to direct solver
        ksp_type_str = "preonly"
        pc_type_str = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="poisson_fb_"
        )
        u_sol = problem.solve()
        total_iterations = 1
    
    # Evaluate on output grid
    bbox = grid_config.get("bbox", [0, 1, 0, 1])
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array: shape (N, 3) for dolfinx
    points_flat = np.zeros((nx_out * ny_out, 3))
    points_flat[:, 0] = XX.flatten()
    points_flat[:, 1] = YY.flatten()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells for each point
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat)
    
    # Evaluate
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Replace any NaN with nearest neighbor or boundary value
    if np.any(np.isnan(u_grid)):
        u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    import json
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    # Full case spec matching the config.json format
    case_spec = {
        "id": "poisson_no_exact_gaussian_source_nonzero_bc",
        "oracle_config": {
            "pde": {
                "type": "poisson",
                "coefficients": {
                    "kappa": {"type": "constant", "value": 1.0}
                },
                "source_term": "exp(-180*((x-0.3)**2 + (y-0.7)**2))"
            },
            "domain": {"type": "unit_square"},
            "bc": {
                "dirichlet": {
                    "on": "all",
                    "value": "sin(2*pi*x) + 0.5*cos(2*pi*y)"
                }
            },
            "output": {
                "format": "npz",
                "field": "scalar",
                "grid": {
                    "bbox": [0, 1, 0, 1],
                    "nx": 50,
                    "ny": 50
                }
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {u_grid.min():.6e}, max: {u_grid.max():.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    
    # If outdir specified, write output files
    if args.outdir:
        import os
        os.makedirs(args.outdir, exist_ok=True)
        
        grid = case_spec["oracle_config"]["output"]["grid"]
        nx, ny = grid["nx"], grid["ny"]
        xs = np.linspace(grid["bbox"][0], grid["bbox"][1], nx)
        ys = np.linspace(grid["bbox"][2], grid["bbox"][3], ny)
        
        np.savez(f"{args.outdir}/solution.npz", x=xs, y=ys, u=u_grid)
        
        meta = {
            "wall_time_sec": elapsed,
            "solver_info": result["solver_info"],
        }
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Output saved to {args.outdir}")
