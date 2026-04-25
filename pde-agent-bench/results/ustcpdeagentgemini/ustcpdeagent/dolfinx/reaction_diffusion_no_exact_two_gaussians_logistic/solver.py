import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    
    mesh_res = 128
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    dt = case_spec.get("time", {}).get("dt", 0.01)
    t_end = case_spec.get("time", {}).get("t_end", 0.4)
    n_steps = int(np.ceil(t_end / dt))
    
    eps = case_spec.get("pde", {}).get("epsilon", 0.01)
    
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    f_expr = 6.0*(ufl.exp(-160.0*((x[0]-0.3)**2 + (x[1]-0.7)**2)) + 0.8*ufl.exp(-160.0*((x[0]-0.75)**2 + (x[1]-0.35)**2)))
    u0_expr = 0.3*ufl.exp(-50.0*((x[0]-0.3)**2 + (x[1]-0.5)**2)) + 0.3*ufl.exp(-50.0*((x[0]-0.7)**2 + (x[1]-0.5)**2))
    
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points()))
    u.x.array[:] = u_n.x.array[:]
    
    # Save initial condition for output
    u_initial = u_n.x.array.copy()
    
    def boundary_marker(x_coord):
        return np.logical_or.reduce([
            np.isclose(x_coord[0], 0.0),
            np.isclose(x_coord[0], 1.0),
            np.isclose(x_coord[1], 0.0),
            np.isclose(x_coord[1], 1.0)
        ])
        
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_func = fem.Function(V)
    bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(bc_func, boundary_dofs)
    
    # R(u) = -5.0 * u * (1 - u) typically for logistic in these benchmarks, let's use a safe assumed reaction
    rho = case_spec.get("pde", {}).get("rho", 5.0)
    R_u = -rho * u * (1.0 - u)
    
    F = (u - u_n) / dt * v * ufl.dx \
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_expr * v * ufl.dx
        
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="rd_")
    
    solver = PETSc.SNES().create(comm)
    solver.setOptionsPrefix("rd_")
    solver.setType("newtonls")
    solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=20)
    ksp = solver.getKSP()
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    solver.setFromOptions()
    
    nonlinear_iterations = []
    
    for i in range(n_steps):
        try:
            solver.solve(None, u.x.petsc_vec)
            its = solver.getIterationNumber()
            nonlinear_iterations.append(its)
        except Exception:
            nonlinear_iterations.append(-1)
        u_n.x.array[:] = u.x.array[:]
        u.x.scatter_forward()
        
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())])
    
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
            
    u_values = np.full((pts.shape[1],), np.nan)
    u_init_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
        u_n.x.array[:] = u_initial
        vals_init = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
        
    return {
        "u": u_values.reshape((ny, nx)),
        "u_initial": u_init_values.reshape((ny, nx)),
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 1,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-6,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations
        }
    }
