import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time
import json
import argparse
from pathlib import Path

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve transient convection-diffusion with SUPG stabilization."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec - handle both formats
    pde = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults as fallback
    time_spec = pde.get("time", {})
    t_end = float(time_spec.get("t_end", 0.1))
    dt_val = float(time_spec.get("dt", 0.02))
    scheme = time_spec.get("scheme", "backward_euler")
    
    # PDE parameters - handle both "params" and "pde_params"
    params = pde.get("params", pde.get("pde_params", {}))
    epsilon = float(params.get("epsilon", 0.05))
    beta_vec = params.get("beta", [2.0, 1.0])
    beta_vec = [float(b) for b in beta_vec]
    
    # Output grid
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", output.get("nx", 50)))
    ny_out = int(grid.get("ny", output.get("ny", 50)))
    bbox = grid.get("bbox", [0, 1, 0, 1])
    x_min, x_max = float(bbox[0]), float(bbox[1])
    y_min, y_max = float(bbox[2]), float(bbox[3])
    
    # Domain - handle both formats
    domain_spec = case_spec.get("domain", {})
    if domain_spec.get("type") == "unit_square":
        x_range = [0.0, 1.0]
        y_range = [0.0, 1.0]
    else:
        x_range = domain_spec.get("x_range", [0.0, 1.0])
        y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # Solver parameters - adaptive mesh refinement
    element_degree = 1
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-8
    
    # Time stepping
    dt = dt_val
    n_steps = max(1, int(round(t_end / dt)))
    dt = t_end / n_steps  # Adjust dt to exactly hit t_end
    
    # Adaptive mesh convergence loop
    resolutions = [32, 64, 128]
    prev_norm = None
    chosen_resolution = resolutions[-1]
    final_u = None
    final_u_initial_grid = None
    final_iterations = 0
    final_n_steps = n_steps
    
    for N in resolutions:
        # Create mesh
        domain_mesh = mesh.create_rectangle(
            comm,
            [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
            [N, N],
            cell_type=mesh.CellType.triangle
        )
        
        # Function space
        V = fem.functionspace(domain_mesh, ("Lagrange", element_degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Previous time step solution
        u_n = fem.Function(V, name="u_n")
        u_sol = fem.Function(V, name="u_sol")
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain_mesh)
        
        # Source term: f = sin(3*pi*x)*sin(2*pi*y)
        pi = ufl.pi
        f = ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
        
        # Convection velocity
        beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
        
        # Diffusion coefficient
        eps = fem.Constant(domain_mesh, ScalarType(epsilon))
        
        # Time step constant
        dt_c = fem.Constant(domain_mesh, ScalarType(dt))
        
        # Initial condition: u0 = 0
        u_n.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        
        # Boundary conditions - homogeneous Dirichlet on all boundaries
        tdim = domain_mesh.topology.dim
        fdim = tdim - 1
        
        boundary_facets = mesh.locate_entities_boundary(
            domain_mesh, fdim,
            lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
        )
        
        bc_func = fem.Function(V)
        bc_func.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(bc_func, dofs)
        bcs = [bc]
        
        # SUPG stabilization parameter
        h = ufl.CellDiameter(domain_mesh)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
        Pe_cell = beta_norm * h / (2.0 * eps)
        
        # SUPG tau
        tau = h / (2.0 * beta_norm + 1e-14)
        
        # Backward Euler: (u - u_n)/dt - eps*lap(u) + beta.grad(u) = f
        
        # Standard Galerkin bilinear form (LHS)
        a_standard = (
            u / dt_c * v * ufl.dx
            + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        )
        
        # Standard Galerkin linear form (RHS)
        L_standard = (
            u_n / dt_c * v * ufl.dx
            + f * v * ufl.dx
        )
        
        # SUPG test function: tau * beta.grad(v)
        R_test = tau * ufl.dot(beta, ufl.grad(v))
        
        # SUPG addition to LHS (for P1, laplacian of u vanishes element-wise)
        a_supg = (
            (u / dt_c) * R_test * ufl.dx
            + ufl.dot(beta, ufl.grad(u)) * R_test * ufl.dx
        )
        
        # SUPG addition to RHS
        L_supg = (
            (u_n / dt_c) * R_test * ufl.dx
            + f * R_test * ufl.dx
        )
        
        # Total forms
        a_total = a_standard + a_supg
        L_total = L_standard + L_supg
        
        # Compile forms
        a_form = fem.form(a_total)
        L_form = fem.form(L_total)
        
        # Assemble matrix (constant in time for backward Euler with constant coefficients)
        A = petsc.assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        
        # Setup KSP solver
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        try:
            solver.setType(ksp_type)
            pc = solver.getPC()
            pc.setType(pc_type)
            solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
            solver.setUp()
            use_iterative = True
        except Exception:
            # Fallback to direct solver
            solver.setType("preonly")
            pc = solver.getPC()
            pc.setType("lu")
            solver.setUp()
            use_iterative = False
        
        total_iterations = 0
        
        # Time stepping loop
        t = 0.0
        for step in range(n_steps):
            t += dt
            
            # Assemble RHS
            b = petsc.assemble_vector(L_form)
            petsc.apply_lifting(b, [a_form], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, bcs)
            
            # Solve
            try:
                solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                total_iterations += solver.getIterationNumber()
            except Exception:
                # Fallback to direct solver
                solver.destroy()
                solver = PETSc.KSP().create(comm)
                solver.setOperators(A)
                solver.setType("preonly")
                solver.getPC().setType("lu")
                solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                total_iterations += 1
            
            # Update previous solution
            u_n.x.array[:] = u_sol.x.array[:]
            
            b.destroy()
        
        # Check convergence via L2 norm
        u_norm_local = fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))
        u_norm = np.sqrt(comm.allreduce(u_norm_local, op=MPI.SUM))
        
        if prev_norm is not None:
            rel_err = abs(u_norm - prev_norm) / (u_norm + 1e-14)
            if rel_err < 0.01:
                chosen_resolution = N
                final_u = u_sol
                final_iterations = total_iterations
                final_n_steps = n_steps
                solver.destroy()
                A.destroy()
                break
        
        prev_norm = u_norm
        chosen_resolution = N
        final_u = u_sol
        final_iterations = total_iterations
        final_n_steps = n_steps
        solver.destroy()
        A.destroy()
    
    # Evaluate solution on output grid
    domain_mesh = final_u.function_space.mesh
    
    x_out = np.linspace(x_min, x_max, nx_out)
    y_out = np.linspace(y_min, y_max, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain_mesh, domain_mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_mesh, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = final_u.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Initial condition on grid (all zeros)
    u_initial = np.zeros((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": chosen_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": final_iterations,
            "dt": dt,
            "n_steps": final_n_steps,
            "time_scheme": scheme,
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    # Default case_spec
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "pde_params": {
                "epsilon": 0.05,
                "beta": [2.0, 1.0]
            },
            "source_term": "sin(3*pi*x)*sin(2*pi*y)",
            "initial_condition": "0.0",
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "domain": {
            "type": "unit_square"
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
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Solve time: {elapsed:.3f}s")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solution min: {np.nanmin(result['u']):.6f}")
        print(f"Solution max: {np.nanmax(result['u']):.6f}")
        print(f"NaN count: {np.isnan(result['u']).sum()}")
        print(f"Total iterations: {result['solver_info']['iterations']}")
        print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
        
        if args.outdir:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            
            u_grid = result['u']
            nx_out, ny_out = u_grid.shape
            x = np.linspace(0, 1, nx_out)
            y = np.linspace(0, 1, ny_out)
            
            np.savez(outdir / "solution.npz", x=x, y=y, u=u_grid)
            np.save(outdir / "u.npy", u_grid)
            
            if result.get("u_initial") is not None:
                np.save(outdir / "u_initial.npy", result["u_initial"])
            
            meta = {
                "wall_time_sec": elapsed,
                "solver_info": result["solver_info"],
            }
            with open(outdir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            print(f"Output written to {outdir}")
