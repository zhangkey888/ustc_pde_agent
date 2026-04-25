import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
import time

def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    t0 = 0.0
    t_end = 0.2
    dt = 0.005
    n_steps = int(np.round((t_end - t0) / dt))
    
    comm = MPI.COMM_WORLD
    mesh_res = 48
    elem_degree = 2
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    epsilon = 0.01
    
    def u_exact_np(x, y, t):
        return np.exp(-t) * 0.25 * np.sin(2*np.pi*x) * np.sin(np.pi*y)
        
    def f_exact_np(x, y, t):
        u = u_exact_np(x, y, t)
        du_dt = -u
        lap_u = np.exp(-t) * 0.25 * ( - (2*np.pi)**2 * np.sin(2*np.pi*x) * np.sin(np.pi*y) - (np.pi)**2 * np.sin(2*np.pi*x) * np.sin(np.pi*y) )
        R = u**3 - u
        return du_dt - epsilon * lap_u + R

    def eval_u_exact(t, pts):
        return u_exact_np(pts[0], pts[1], t)
        
    def eval_f(t, pts):
        return f_exact_np(pts[0], pts[1], t)
    
    u = fem.Function(V)
    u_n = fem.Function(V)
    f = fem.Function(V)
    
    u_n.interpolate(lambda pts: eval_u_exact(t0, pts))
    u.x.array[:] = u_n.x.array[:]
    
    v = ufl.TestFunction(V)
    
    F_expr = (u - u_n) / dt * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u**3 - u) * v * ufl.dx \
        - f * v * ufl.dx
        
    J_expr = ufl.derivative(F_expr, u)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
    
    problem = petsc.NonlinearProblem(F_expr, u, bcs=[bc], J=J_expr,
                                     petsc_options_prefix="ac_",
                                     petsc_options=petsc_options)
    
    nonlinear_iters = []
    
    t = t0
    for step in range(n_steps):
        t += dt
        u_bc.interpolate(lambda pts: eval_u_exact(t, pts))
        f.interpolate(lambda pts: eval_f(t, pts))
        
        problem.solve()
        
        its = problem.solver.getIterationNumber()
        nonlinear_iters.append(its)
        
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_out = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny_out, nx_out))
    u_initial = u_exact_np(XX, YY, t0)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": petsc_options["ksp_type"],
        "pc_type": petsc_options["pc_type"],
        "rtol": petsc_options["snes_rtol"],
        "iterations": sum(nonlinear_iters), 
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial
    }
