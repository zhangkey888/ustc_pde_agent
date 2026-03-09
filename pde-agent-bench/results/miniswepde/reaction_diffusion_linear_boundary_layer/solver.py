import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation with manufactured solution."""
    
    # Extract parameters from case_spec (handles the oracle_config structure)
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde_spec = oracle_config.get("pde", {})
    pde_params = pde_spec.get("pde_params", {})
    
    # Diffusion coefficient
    epsilon = pde_params.get("epsilon", 0.03)
    
    # Reaction coefficient (linear: R(u) = alpha * u)
    reaction_spec = pde_params.get("reaction", {})
    alpha = reaction_spec.get("alpha", 1.0)
    
    # Time parameters
    time_spec = pde_spec.get("time", {})
    t_end = time_spec.get("t_end", 0.3)
    dt = time_spec.get("dt", 0.005)
    time_scheme = time_spec.get("scheme", "backward_euler")
    t0 = time_spec.get("t0", 0.0)
    
    # Output grid
    output_spec = oracle_config.get("output", {})
    grid_spec = output_spec.get("grid", {})
    nx_out = grid_spec.get("nx", 75)
    ny_out = grid_spec.get("ny", 75)
    bbox = grid_spec.get("bbox", [0, 1, 0, 1])
    
    # Mesh and element parameters
    # With epsilon=0.03, we need good resolution for boundary layers
    # Oracle uses 180 with degree 1; let's use similar or degree 2 with less resolution
    mesh_resolution = 128
    element_degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time variable as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Manufactured solution: u_exact = exp(-t) * exp(4*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f from the PDE:
    # du/dt - eps * laplacian(u) + alpha * u = f
    # u = exp(-t) * exp(4x) * sin(pi*y)
    # du/dt = -u
    # d2u/dx2 = 16*u, d2u/dy2 = -pi^2*u
    # laplacian(u) = (16 - pi^2)*u
    # f = -u - eps*(16 - pi^2)*u + alpha*u
    # f = u * (-1 - eps*(16 - pi^2) + alpha)
    pi_val = np.pi
    f_coeff = -1.0 - epsilon * (16.0 - pi_val**2) + alpha
    f_expr = f_coeff * u_exact_ufl
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    
    # Constants for the variational form
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    alpha_const = fem.Constant(domain, PETSc.ScalarType(alpha))
    
    # Backward Euler weak form:
    # (u - u_n)/dt * v + eps * grad(u) . grad(v) + alpha * u * v = f * v
    a = (1.0 / dt_const * ufl.inner(u, v) * ufl.dx 
         + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + alpha_const * ufl.inner(u, v) * ufl.dx)
    
    L = (ufl.inner(f_expr, v) * ufl.dx 
         + 1.0 / dt_const * ufl.inner(u_n, v) * ufl.dx)
    
    # Set initial condition: u(x, 0) = exp(4x)*sin(pi*y)
    u_n.interpolate(lambda x_arr: np.exp(4.0 * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (will be reassembled each step due to BC changes)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc_func, boundary_dofs)])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    # Setup solver - CG is appropriate for symmetric positive definite system
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    # Time stepping
    t = t0
    n_steps = int(np.round((t_end - t0) / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = t
        u_bc_func.interpolate(lambda x_arr, tv=t_val: 
                               np.exp(-tv) * np.exp(4.0 * x_arr[0]) * np.sin(np.pi * x_arr[1]))
        
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Reassemble matrix with updated BCs
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Initial condition on same grid
    u_init_values = np.exp(4.0 * XX) * np.sin(np.pi * YY)
    
    result = {
        "u": u_grid,
        "u_initial": u_init_values,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    
    return result


if __name__ == "__main__":
    import json
    start = time_module.time()
    
    # Load the actual case_spec that the evaluator would use
    case_spec = {
        "id": "reaction_diffusion_linear_boundary_layer",
        "oracle_config": {
            "pde": {
                "type": "reaction_diffusion",
                "pde_params": {
                    "epsilon": 0.03,
                    "reaction": {
                        "type": "linear",
                        "alpha": 1.0
                    }
                },
                "time": {
                    "t0": 0.0,
                    "t_end": 0.3,
                    "dt": 0.005,
                    "scheme": "backward_euler"
                },
                "manufactured_solution": {
                    "u": "exp(-t)*(exp(4*x)*sin(pi*y))"
                }
            },
            "domain": {"type": "unit_square"},
            "output": {
                "grid": {
                    "bbox": [0, 1, 0, 1],
                    "nx": 75,
                    "ny": 75
                }
            }
        }
    }
    
    result = solve(case_spec)
    elapsed = time_module.time() - start
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    
    # Check exact solution
    nx, ny = 75, 75
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.3) * np.exp(4.0 * XX) * np.sin(np.pi * YY)
    print(f"Exact range: [{np.nanmin(u_exact):.6f}, {np.nanmax(u_exact):.6f}]")
    
    # RMS error
    error_rms = np.sqrt(np.mean((result['u'] - u_exact)**2))
    print(f"RMS error: {error_rms:.6e}")
    
    # Relative L2 error (what the evaluator uses)
    rel_l2 = np.sqrt(np.sum((result['u'] - u_exact)**2)) / np.sqrt(np.sum(u_exact**2))
    print(f"Relative L2 error: {rel_l2:.6e}")
    print(f"Target: <= 4.81e-03")
    print(f"PASS: {rel_l2 <= 4.81e-3}")
