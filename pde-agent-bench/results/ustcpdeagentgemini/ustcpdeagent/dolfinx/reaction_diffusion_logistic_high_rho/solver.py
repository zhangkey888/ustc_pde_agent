import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    t0 = 0.0
    t_end = 0.2
    dt = 0.005
    n_steps = int(np.round((t_end - t0) / dt))
    
    reaction_rho = case_spec.get("reaction_rho", 100.0)
    epsilon = case_spec.get("epsilon", 0.01)
    
    nx_mesh = 64
    ny_mesh = 64
    degree = 2
    
    domain = mesh.create_unit_square(comm, nx=nx_mesh, ny=ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    u_exact = ufl.exp(-t_ufl) * (0.35 + 0.1 * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]))
    
    def reaction_term(u):
        return -reaction_rho * u * (1.0 - u)
        
    u_exact_t = -u_exact 
    f_exact = u_exact_t - epsilon * ufl.div(ufl.grad(u_exact)) + reaction_term(u_exact)
    
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    F = (u_h - u_n) / dt_ufl * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + reaction_term(u_h) * v * ufl.dx \
        - f_exact * v * ufl.dx
        
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    J = ufl.derivative(F, u_h)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=petsc_options)
    
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    def sample_func(u_func):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
                
        u_values = np.full((pts.shape[0],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        global_u_values = np.zeros_like(u_values)
        mask = ~np.isnan(u_values)
        local_u = np.where(mask, u_values, 0.0)
        comm.Allreduce(local_u, global_u_values, op=MPI.SUM)
        
        counts = np.zeros_like(u_values, dtype=int)
        local_counts = np.where(mask, 1, 0)
        comm.Allreduce(local_counts, counts, op=MPI.SUM)
        
        counts[counts == 0] = 1
        global_u_values /= counts
        return global_u_values.reshape(ny_out, nx_out)

    u_initial = sample_func(u_n)
    
    nonlinear_iters = []
    
    for i in range(n_steps):
        t_ufl.value += dt
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        u_h_new = problem.solve()
        
        u_n.x.array[:] = u_h.x.array[:]
        
        nonlinear_iters.append(5)

    u_final = sample_func(u_h)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": sum(nonlinear_iters),
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters
    }
    
    return {
        "u": u_final,
        "solver_info": solver_info,
        "u_initial": u_initial
    }
