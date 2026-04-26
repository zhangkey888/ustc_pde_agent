import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    wall_start = time.time()
    
    # Extract PDE parameters
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    epsilon = float(pde.get("epsilon", 1.0))
    alpha = float(pde.get("alpha", 1.0))
    
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.5))
    dt = float(time_params.get("dt", 0.01))
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Output grid
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # High accuracy parameters - utilize time budget
    mesh_res = 128
    element_degree = 3
    dt_val = min(dt, 0.0025)  # smaller time step for better accuracy
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    n_steps = int(round((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps  # exact dt
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n = fem.Function(V)
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))
    
    # Source term: f = (-1 + eps*2*pi^2 + alpha)*exp(-t)*sin(pi*x)*sin(pi*y)
    f_func = fem.Function(V)
    source_coeff = ScalarType(-1.0 + epsilon * 2.0 * np.pi**2 + alpha)
    f_ufl = source_coeff * ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    
    # BC: u=0 on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    # Variational form (backward Euler)
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (1.0/dt_val) * ufl.inner(u_trial, v) * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx \
        + alpha * ufl.inner(u_trial, v) * ufl.dx
    
    L = (1.0/dt_val) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_func, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.getPC().setHYPREType("boomeramg")
    
    b = petsc.create_vector(L_form.function_spaces)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    
    for step in range(n_steps):
        current_time = t0 + (step + 1) * dt_val
        t_const.value = ScalarType(current_time)
        f_func.interpolate(f_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        its = solver.getIterationNumber()
        total_iterations += its
        
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))
    
    u_init_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)
    
    # L2 error verification
    u_exact = fem.Function(V)
    u_exact_expr_ufl = ufl.exp(-ScalarType(t_end)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact.interpolate(fem.Expression(u_exact_expr_ufl, V.element.interpolation_points))
    
    error_sq = fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx))
    l2_error = np.sqrt(domain.comm.allreduce(error_sq, op=MPI.SUM))
    
    wall_end = time.time()
    if domain.comm.rank == 0:
        print(f"L2 error at t={t_end}: {l2_error:.6e}")
        print(f"Wall time: {wall_end - wall_start:.3f}s")
        print(f"Mesh: {mesh_res}, Degree: {element_degree}, dt: {dt_val}, Steps: {n_steps}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 1.0,
            "alpha": 1.0,
            "reaction": "linear",
            "time": {
                "t0": 0.0,
                "t_end": 0.5,
                "dt": 0.01,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(case_spec)
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
