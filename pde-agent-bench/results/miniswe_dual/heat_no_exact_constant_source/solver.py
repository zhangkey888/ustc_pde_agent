import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the heat equation (transient) using backward Euler."""
    
    # Extract parameters from case_spec with defaults
    pde = case_spec.get("pde", {})
    
    # Also check oracle_config path
    oracle_config = case_spec.get("oracle_config", {})
    if not pde and oracle_config:
        pde = oracle_config.get("pde", {})
    
    coeffs = pde.get("coefficients", {})
    if isinstance(coeffs.get("kappa"), dict):
        kappa_val = float(coeffs["kappa"]["value"])
    else:
        kappa_val = float(coeffs.get("kappa", 1.0))
    
    # Source term - check multiple possible keys
    source = pde.get("source_term", pde.get("source", 1.0))
    if isinstance(source, dict):
        source = float(source.get("value", 1.0))
    else:
        source = float(source)
    
    # Initial condition
    ic_val = pde.get("initial_condition", 0.0)
    if isinstance(ic_val, dict):
        ic_val = float(ic_val.get("value", 0.0))
    else:
        ic_val = float(ic_val)
    
    # Time parameters - hardcoded defaults as fallback
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1)) if time_params else 0.1
    dt = float(time_params.get("dt", 0.02)) if time_params else 0.02
    scheme = time_params.get("scheme", "backward_euler") if time_params else "backward_euler"
    
    # Boundary condition value
    bc_spec = pde.get("boundary_conditions", {})
    # Also check oracle_config bc
    if not bc_spec:
        bc_spec = oracle_config.get("bc", {}).get("dirichlet", {})
    g_val = 0.0
    if isinstance(bc_spec, dict):
        val = bc_spec.get("value", 0.0)
        if isinstance(val, str):
            g_val = float(val)
        else:
            g_val = float(val)
    elif isinstance(bc_spec, list) and len(bc_spec) > 0:
        g_val = float(bc_spec[0].get("value", 0.0))
    
    # Output grid spec
    output_spec = oracle_config.get("output", {}).get("grid", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)
    bbox = output_spec.get("bbox", [0, 1, 0, 1])
    
    # Mesh resolution - moderate is fine for this problem
    N = 32
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution at current and previous time step
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda x: np.full(x.shape[1], ic_val))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.full(x.shape[1], ic_val))
    
    # Current solution
    u_h = fem.Function(V, name="u_h")
    
    # Compute actual dt and n_steps
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    
    # Constants
    dt_const = fem.Constant(domain, ScalarType(actual_dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa_val))
    f_const = fem.Constant(domain, ScalarType(source))
    
    # Backward Euler weak form
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (f_const * v) * ufl.dx + (u_n * v / dt_const) * ufl.dx
    
    # Boundary conditions: u = g on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(g_val), dofs, V)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant for backward Euler with constant coefficients)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b_vec = petsc.create_vector(V)
    
    # Setup solver
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type_str)
    pc = solver.getPC()
    pc.setType(pc_type_str)
    solver.setTolerances(rtol=rtol_val)
    solver.setUp()
    
    # Time stepping
    total_iterations = 0
    
    for step in range(n_steps):
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on output grid
    x_coords = np.linspace(bbox[0], bbox[1], nx_out)
    y_coords = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    
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
        
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import argparse
    import json
    import time
    
    parser = argparse.ArgumentParser(description='Heat equation solver')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Hardcoded case_spec for this specific problem
    case_spec = {
        "pde": {
            "type": "heat",
            "source_term": "1.0",
            "initial_condition": "0.0",
            "coefficients": {"kappa": {"type": "constant", "value": 1.0}},
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        },
        "oracle_config": {
            "pde": {
                "type": "heat",
                "coefficients": {"kappa": {"type": "constant", "value": 1.0}},
                "source_term": "1.0",
                "initial_condition": "0.0",
                "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
            },
            "domain": {"type": "unit_square"},
            "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
            "output": {
                "format": "npz",
                "field": "scalar",
                "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50}
            },
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    
    if args.outdir:
        import os
        os.makedirs(args.outdir, exist_ok=True)
        
        # Get grid coordinates
        nx_out, ny_out = u_grid.shape
        x = np.linspace(0, 1, nx_out)
        y = np.linspace(0, 1, ny_out)
        
        np.savez(f"{args.outdir}/solution.npz", x=x, y=y, u=u_grid)
        np.save(f"{args.outdir}/u.npy", u_grid)
        
        if result.get("u_initial") is not None:
            np.save(f"{args.outdir}/u_initial.npy", result["u_initial"])
        
        meta = {
            "wall_time_sec": elapsed,
            "solver_info": result["solver_info"],
        }
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Solution saved to {args.outdir}")
    
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {np.nanmin(u_grid):.6f}, max: {np.nanmax(u_grid):.6f}")
    print(f"Solution mean: {np.nanmean(u_grid):.6f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
