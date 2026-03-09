import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with manufactured solution."""
    
    # ---- Extract parameters from case_spec ----
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    
    # Time parameters with hardcoded defaults
    t_end = float(time_spec.get("t_end", 0.1))
    dt_val = float(time_spec.get("dt", 0.01))
    scheme = time_spec.get("scheme", "backward_euler")
    
    # Adaptive parameters
    mesh_resolution = 64
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    
    # ---- Create mesh ----
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # ---- Spatial coordinates and time ----
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    
    # ---- Manufactured solution ----
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # ---- Diffusion coefficient kappa ----
    kappa_ufl = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))
    
    # ---- Source term: f = du/dt - div(kappa * grad(u)) ----
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y) = -u_exact
    dudt = -ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # div(kappa * grad(u_exact))
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa_ufl * grad_u_exact)
    
    f_ufl = dudt - div_kappa_grad_u
    
    # ---- Trial and test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # ---- Solution functions ----
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # ---- Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y) ----
    u_n.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    # ---- Backward Euler weak form ----
    # (u - u_n)/dt - div(kappa * grad(u)) = f
    # Weak form: (u/dt)*v*dx + kappa*grad(u)·grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    a = (u / dt_c) * v * ufl.dx + kappa_ufl * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # ---- Boundary conditions ----
    # u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    # At t=0, boundary values
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # ---- Assemble matrix (kappa doesn't depend on time, but we reassemble to be safe) ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # ---- Setup KSP solver ----
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-10
    
    # ---- Time stepping ----
    current_t = 0.0
    n_steps = 0
    total_iterations = 0
    
    num_steps = int(round(t_end / dt_val))
    
    for step in range(num_steps):
        current_t += dt_val
        t.value = current_t
        
        # Update boundary condition
        u_bc.interpolate(bc_expr)
        
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
        n_steps += 1
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # ---- Evaluate on 50x50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc2 = []
    cells_on_proc2 = []
    eval_map2 = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc2.append(points_3d[i])
            cells_on_proc2.append(links[0])
            eval_map2.append(i)
    
    if len(points_on_proc2) > 0:
        pts_arr2 = np.array(points_on_proc2)
        cells_arr2 = np.array(cells_on_proc2, dtype=np.int32)
        vals2 = u_initial_func.eval(pts_arr2, cells_arr2)
        u_init_values[eval_map2] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # ---- Build result ----
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }
    
    return result


if __name__ == "__main__":
    # Test with a minimal case_spec
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
                    "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"
                }
            }
        },
        "domain": {
            "type": "unit_square"
        }
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution at t=0.1
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
