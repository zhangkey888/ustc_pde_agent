import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    mesh_res = 128
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    t = 0.0
    t_end = 0.1
    dt = 0.01 
    num_steps = int(np.ceil(t_end / dt))
    
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Function(V)
    f.interpolate(lambda x: np.sin(np.pi*x[0]) * np.cos(np.pi*x[1]))
    
    kappa = 1.0
    F = (u - u_n) / dt * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-8}
    )
    
    total_iters = 0
    u_sol = fem.Function(V)
    
    for i in range(num_steps):
        t += dt
        sol = problem.solve()
        u_sol.x.array[:] = sol.x.array[:]
        u_n.x.array[:] = u_sol.x.array[:]
        total_iters += problem.solver.getIterationNumber()
        
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
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
            
    u_values = np.zeros(pts.shape[0])
    u_init_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
        
    u_grid = u_values.reshape(ny_out, nx_out)
    u_init_grid = u_init_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "rtol": 1e-8,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }
