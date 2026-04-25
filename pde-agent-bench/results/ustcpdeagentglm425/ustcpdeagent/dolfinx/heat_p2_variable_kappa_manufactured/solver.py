import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde = case_spec["pde"]
    domain_info = pde["domain"]
    time_info = pde["time"]
    output_info = case_spec["output"]
    grid_info = output_info["grid"]
    
    # Domain
    bbox = domain_info["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Time parameters
    t0 = float(time_info["t0"])
    t_end = float(time_info["t_end"])
    dt_suggested = float(time_info["dt"])
    
    # Output grid
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox_out = grid_info["bbox"]
    
    # Choose numerical parameters
    mesh_res = 128
    element_degree = 2
    dt = dt_suggested  # 0.01
    n_steps = int(round((t_end - t0) / dt))  # 6 steps
    
    # Create mesh
    p0 = np.array([xmin, ymin], dtype=np.float64)
    p1 = np.array([xmax, ymax], dtype=np.float64)
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                    cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Variable coefficient kappa = 1 + 0.4*sin(2*pi*x)*sin(2*pi*y)
    kappa_ufl = 1.0 + 0.4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_ex_ufl = ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # du/dt = -u_ex
    du_dt_ufl = -u_ex_ufl
    
    # div(kappa * grad(u_ex))
    div_kappa_grad_u = ufl.div(kappa_ufl * ufl.grad(u_ex_ufl))
    
    # Source term: f = du/dt - div(kappa * grad(u))
    f_ufl = du_dt_ufl - div_kappa_grad_u
    
    # Boundary condition function
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    # Locate boundary DOFs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    t_const.value = PETSc.ScalarType(t0)
    u_n.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))
    
    # Variational form (backward Euler)
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(u_trial, v_test) * ufl.dx + dt * ufl.inner(kappa_ufl * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = ufl.inner(u_n, v_test) * ufl.dx + dt * ufl.inner(f_ufl, v_test) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant since kappa doesn't depend on time)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    rtol = 1e-10
    solver.setTolerances(rtol=rtol)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    
    total_iterations = 0
    
    # Build point evaluation structures once
    xs_out = np.linspace(bbox_out[0], bbox_out[1], nx_out)
    ys_out = np.linspace(bbox_out[2], bbox_out[3], ny_out)
    XX, YY = np.meshgrid(xs_out, ys_out)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    pts_on_proc_arr = np.array(points_on_proc) if points_on_proc else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if cells_on_proc else np.empty(0, dtype=np.int32)
    
    def eval_on_grid(fem_func):
        u_values = np.full((pts.shape[0],), np.nan)
        if len(points_on_proc) > 0:
            vals = fem_func.eval(pts_on_proc_arr, cells_on_proc_arr)
            u_values[eval_map] = vals.flatten()
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=MPI.SUM)
        return u_global.reshape(ny_out, nx_out)
    
    # Compute initial condition grid
    u_initial_grid = eval_on_grid(u_n)
    
    # Time stepping
    for n in range(n_steps):
        t_current = t0 + (n + 1) * dt
        t_const.value = PETSc.ScalarType(t_current)
        
        # Update boundary condition
        u_bc_func.interpolate(u_bc_expr)
        
        # Assemble RHS
        b = petsc.assemble_vector(L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Track iterations
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Compute L2 error for verification
    u_ex_final = fem.Function(V)
    u_ex_final.interpolate(u_bc_expr)
    error_form = fem.form(ufl.inner(u_sol - u_ex_final, u_sol - u_ex_final) * ufl.dx)
    l2_error_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(domain.comm.allreduce(l2_error_sq, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {l2_error:.6e}")
        print(f"Total KSP iterations: {total_iterations}, n_steps={n_steps}")
    
    # Sample solution on output grid
    u_grid = eval_on_grid(u_sol)
    
    # Get solver info
    ksp_type = str(solver.getType())
    pc_type = str(solver.getPC().getType())
    
    # Destroy PETSc objects
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "heat",
            "domain": {"type": "rectangle", "bbox": [0.0, 1.0, 0.0, 1.0]},
            "time": {"t0": 0.0, "t_end": 0.06, "dt": 0.01, "scheme": "backward_euler"},
            "coefficients": {"kappa": {"type": "expr", "expr": "1 + 0.4*sin(2*pi*x)*sin(2*pi*y)"}},
        },
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    
    import time
    t_start = time.time()
    result = solve(case_spec)
    t_elapsed = time.time() - t_start
    
    print(f"Wall time: {t_elapsed:.3f} s")
    print(f"u shape: {result["u"].shape}")
    print(f"u min/max: {result["u"].min():.6e} / {result["u"].max():.6e}")
    print(f"Solver info: {result["solver_info"]}")
