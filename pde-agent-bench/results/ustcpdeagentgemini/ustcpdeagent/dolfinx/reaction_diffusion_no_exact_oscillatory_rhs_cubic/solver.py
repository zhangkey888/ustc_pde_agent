import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # PDE parameters
    pde_spec = case_spec.get("pde", {})
    epsilon_val = pde_spec.get("epsilon", 0.01)
    
    time_spec = pde_spec.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.3)
    dt_val = time_spec.get("dt", 0.005)
    
    # Resolution and element degree
    mesh_res = 128
    degree = 2
    
    msh = mesh.create_rectangle(comm, [np.array([xmin, ymin]), np.array([xmax, ymax])], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(6*ufl.pi*x[0]) * ufl.sin(5*ufl.pi*x[1])
    u0_expr = 0.2 * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u.x.array[:] = u_n.x.array[:]
    
    # Point sampling setup
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_init_vals = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals.flatten()
    u_initial = u_init_vals.reshape((ny_out, nx_out))
    
    # Boundary condition
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # Variational form
    dt = fem.Constant(msh, PETSc.ScalarType(dt_val))
    epsilon = fem.Constant(msh, PETSc.ScalarType(epsilon_val))
    
    F = ((u - u_n) / dt) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + (u**3) * v * ufl.dx - f_expr * v * ufl.dx
    J = ufl.derivative(F, u)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-8,
        "snes_max_it": 50,
        "ksp_type": "cg",
        "pc_type": "ilu"
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=petsc_options)
    
    t = t0
    n_steps = int(np.round((t_end - t0) / dt_val))
    
    total_lin_iters = 0
    nonlin_iters_list = []
    
    for step in range(n_steps):
        t += dt_val
        problem.solve()
        
        n_iters = problem.solver.getIterationNumber()
        lin_iters = problem.solver.getLinearSolveIterations()
        
        nonlin_iters_list.append(n_iters)
        total_lin_iters += lin_iters
        
        u_n.x.array[:] = u.x.array[:]
        
    u_sol_vals = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_sol_vals[eval_map] = vals.flatten()
    u_out = u_sol_vals.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_lin_iters,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlin_iters_list
    }
    
    return {
        "u": u_out,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    pass

