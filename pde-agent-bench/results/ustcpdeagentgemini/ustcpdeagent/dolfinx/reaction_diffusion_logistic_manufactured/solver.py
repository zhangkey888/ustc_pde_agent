import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    t0 = 0.0
    t_end = 0.3
    dt_val = 0.01
    num_steps = int(np.round((t_end - t0) / dt_val))
    
    epsilon = 0.1
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    def u_exact_expr(time_val):
        return ufl.exp(-time_val) * (0.2 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    def R(u):
        return u * (1.0 - u)
    
    u_ex = u_exact_expr(t)
    du_dt_ex = -u_ex 
    f = du_dt_ex - epsilon * ufl.div(ufl.grad(u_ex)) + R(u_ex)
    
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    init_expr = fem.Expression(u_exact_expr(PETSc.ScalarType(t0)), V.element.interpolation_points)
    u.interpolate(init_expr)
    u_n.x.array[:] = u.x.array[:]
    
    u_initial = u.x.array.copy()
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    def update_bc(time_val):
        bc_expr = fem.Expression(u_exact_expr(PETSc.ScalarType(time_val)), V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
    
    update_bc(t0 + dt_val)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    F = (u - u_n)/dt * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + R(u)*v * ufl.dx - f*v * ufl.dx
    J = ufl.derivative(F, u)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J,
                                     petsc_options_prefix="rd_",
                                     petsc_options=petsc_options)
                                     
    nonlinear_iterations = []
    linear_iterations = 0
    current_t = t0
    
    for i in range(num_steps):
        current_t += dt_val
        t.value = current_t
        update_bc(current_t)
        
        problem.solve()
        
        num_its = problem.solver.getIterationNumber()
        lin_its = problem.solver.getLinearSolveIterations()
        nonlinear_iterations.append(num_its)
        linear_iterations += lin_its
        
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
        
    u_grid = u_vals.reshape(ny_out, nx_out)
    
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_initial
    u_initial_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_initial_vals[eval_map] = vals_init.flatten()
    u_init_grid = u_initial_vals.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": linear_iterations,
        "dt": dt_val,
        "n_steps": num_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }
