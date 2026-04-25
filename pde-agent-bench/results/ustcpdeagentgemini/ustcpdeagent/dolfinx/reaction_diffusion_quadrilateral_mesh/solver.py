import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Solver parameters
    mesh_res = 64
    degree = 2
    dt = 0.01
    t0 = 0.0
    t_end = 0.4
    
    # Create quadrilateral mesh
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define exact solution
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    
    u_ex = ufl.exp(-t) * (ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # Equation parameters
    epsilon = 1.0
    def R(u):
        return u**3
    
    # Compute source term f analytically
    f = ufl.diff(u_ex, t) - epsilon * ufl.div(ufl.grad(u_ex)) + R(u_ex)
    
    # Trial and Test functions for nonlinear problem
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    # Initial condition
    expr_u0 = fem.Expression(u_ex, V.element.interpolation_points())
    u_n.interpolate(expr_u0)
    u.x.array[:] = u_n.x.array[:]
    
    u_initial_out = None
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(expr_u0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Weak form (Backward Euler)
    F = ufl.inner(u - u_n, v) / dt * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + R(u) * v * ufl.dx \
        - f * v * ufl.dx
        
    J = ufl.derivative(F, u)
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_")
    
    solver = PETSc.SNES().create(comm)
    solver.setOptionsPrefix("rd_")
    solver.setType("newtonls")
    ksp = solver.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    solver.setFromOptions()
    solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=20)
    
    # Time stepping
    current_t = t0
    n_steps = int(np.round((t_end - t0) / dt))
    
    nonlinear_iters = []
    
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        
        # Update BC
        u_bc.interpolate(expr_u0)
        
        # Solve
        solver.setFunction(problem.F, problem.b)
        solver.setJacobian(problem.J, problem.a)
        solver.solve(None, u.x.petsc_vec)
        u.x.scatter_forward()
        
        iters = solver.getIterationNumber()
        nonlinear_iters.append(iters)
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
        
        if step == 0:
            # Sample initial
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
            
            u_values = np.full((pts.shape[0],), np.nan)
            if len(points_on_proc) > 0:
                vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
                u_values[eval_map] = vals.flatten()
            u_initial_out = u_values.reshape(ny_out, nx_out)
            
    # Sample final solution
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
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_out = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_out,
        "u_initial": u_initial_out,
        "solver_info": {
            "mesh_resolution": mesh_res,
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
    }
