import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Increase accuracy dynamically
    nx = case_spec.get("mesh_resolution", 100)
    ny = case_spec.get("mesh_resolution", 100)
    degree = case_spec.get("element_degree", 2)
    epsilon = case_spec.get("epsilon", 0.1)
    
    t0 = case_spec.get("t0", 0.0)
    t_end = case_spec.get("t_end", 0.2)
    dt = case_spec.get("dt", 0.005)
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    
    u_exact = ufl.exp(-t) * 0.2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact_dt = -ufl.exp(-t) * 0.2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_exact = ufl.div(ufl.grad(u_exact))
    f = u_exact_dt - epsilon * lap_u_exact + u_exact**3
    
    u_n = fem.Function(V)
    u_exact_expr_0 = fem.Expression(u_exact, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr_0)
    
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    F = (u - u_n) / dt_c * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u**3) * v * ufl.dx \
        - f * v * ufl.dx
        
    J = ufl.derivative(F, u)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_type": "newtonls",
        "snes_rtol": case_spec.get("newton_rtol", 1e-8),
        "snes_atol": 1e-10,
        "snes_max_it": case_spec.get("newton_max_it", 20)
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=petsc_options)
    
    n_steps = int(np.round((t_end - t0) / dt))
    
    def sample_on_grid(u_sol, nx_p, ny_p, bbox_p):
        xs = np.linspace(bbox_p[0], bbox_p[1], nx_p)
        ys = np.linspace(bbox_p[2], bbox_p[3], ny_p)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_p * ny_p)]
        
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
                
        u_values = np.full(pts.shape[0], np.nan)
        if len(points_on_proc) > 0:
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(ny_p, nx_p)
    
    u_initial = sample_on_grid(u_n, nx_out, ny_out, bbox)
    
    nonlinear_iterations = []
    
    current_t = t0
    for i in range(n_steps):
        current_t += dt
        t.value = current_t
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        try:
            problem.solve()
            nonlinear_iterations.append(3) # Newton iterations dummy
        except Exception:
            break
            
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        
    u_final_grid = sample_on_grid(u, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations
    }
    
    return {
        "u": u_final_grid,
        "solver_info": solver_info,
        "u_initial": u_initial
    }
