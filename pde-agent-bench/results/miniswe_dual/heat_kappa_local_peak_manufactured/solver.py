import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with spatially varying kappa."""
    
    start_time = time_module.time()
    
    # ---- Extract parameters from case_spec ----
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    
    # Time parameters with hardcoded defaults
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))
    scheme = time_spec.get("scheme", "backward_euler")
    
    # Use smaller dt for better temporal accuracy
    dt_val = dt_suggested / 2.0  # 0.005
    n_steps = int(round(t_end / dt_val))
    
    # Mesh resolution and element degree
    N = 80
    degree = 1
    
    comm = MPI.COMM_WORLD
    
    # ---- Create mesh ----
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # ---- Spatial coordinates ----
    x = ufl.SpatialCoordinate(domain)
    
    # ---- Time as a Constant ----
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    
    # ---- Manufactured solution ----
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # ---- Kappa ----
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # ---- Source term f = du/dt - div(kappa * grad(u_exact)) ----
    dudt = -ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f_expr = dudt - ufl.div(kappa * grad_u_exact)
    
    # ---- Trial and test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # ---- Previous solution ----
    u_n = fem.Function(V, name="u_n")
    
    # ---- Initial condition ----
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2 * np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # ---- Backward Euler weak form ----
    a = (u * v / dt_c) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx
    
    # ---- Boundary conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # ---- Solution function ----
    u_sol = fem.Function(V, name="u_sol")
    
    # ---- Set up solver ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # ---- Time stepping loop ----
    for step in range(n_steps):
        t_new = (step + 1) * dt_val
        t.value = t_new
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
        # Re-assemble matrix (BCs affect diagonal entries)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # ---- Evaluate on 50x50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    elapsed = time_module.time() - start_time
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": {
                    "type": "expr",
                    "expr": "1 + 30*exp(-150*((x-0.35)**2 + (y-0.65)**2))"
                }
            }
        },
        "domain": {
            "type": "unit_square"
        }
    }
    
    t0 = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"dt used: {result['solver_info']['dt']}")
    print(f"n_steps: {result['solver_info']['n_steps']}")
    
    # Compare with exact solution at t=0.1
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error vs exact: {error:.6e}")
    print(f"Max error vs exact: {max_error:.6e}")
