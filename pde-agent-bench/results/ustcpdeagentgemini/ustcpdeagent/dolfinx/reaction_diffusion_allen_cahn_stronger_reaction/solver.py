import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import NonlinearProblem

def solve(case_spec: dict) -> dict:
    # 1. Parse case parameters
    pde_spec = case_spec.get("pde", {})
    epsilon = pde_spec.get("epsilon", 1.0)
    lam = pde_spec.get("reaction_lambda", 1.0)
    
    t0 = 0.0
    t_end = 0.1
    dt = pde_spec.get("dt", 0.002)
    
    out_grid = case_spec.get("output", {}).get("grid", {})
    nx_out = out_grid.get("nx", 50)
    ny_out = out_grid.get("ny", 50)
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # 2. Setup mesh
    mesh_res = pde_spec.get("mesh_resolution", 64)
    degree = pde_spec.get("element_degree", 2)
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Define the exact solution (for BC, IC, and source term)
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution expression
    # u = exp(-t)*(0.15 + 0.12*sin(2*pi*x)*sin(2*pi*y))
    u_exact = ufl.exp(-t) * (0.15 + 0.12 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1]))
    
    # R(u) Allen-Cahn
    # R(u) = lam * (u^3 - u)
    def R(u):
        return lam * (u**3 - u)
        
    # Source term
    u_t_exact = -u_exact # derivative of u_exact w.r.t t
    lap_u_exact = ufl.div(ufl.grad(u_exact))
    f_exact = u_t_exact - epsilon * lap_u_exact + R(u_exact)
    
    # 4. Define boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.full(X.shape[1], True))
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, dofs_bc)
    
    # 5. Define initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    
    u_initial = u_n.copy()
    
    # 6. Variational problem
    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    # Backward Euler
    F = (uh - u_n) / dt_c * v * ufl.dx + epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + R(uh) * v * ufl.dx - f_exact * v * ufl.dx
    
    J = ufl.derivative(F, uh)
    
    problem = NonlinearProblem(F, uh, bcs=[bc], J=J, petsc_options_prefix="rd_")
    
    solver = PETSc.SNES().create(comm)
    solver.setOptionsPrefix("rd_")
    solver.setType("newtonls")
    ksp = solver.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    solver.setFromOptions()
    
    # 7. Time loop
    time = t0
    n_steps = 0
    nl_iters = []
    
    while time < t_end - 1e-8:
        time += dt
        t.value = time
        
        # update BC
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
        
        solver.solve(None, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        its = solver.getIterationNumber()
        nl_iters.append(its)
        
        u_n.x.array[:] = uh.x.array[:]
        n_steps += 1
        
    # 8. Evaluation
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
    for i in range(len(pts)):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pts[i])
            cells.append(colliding.links(i)[0])
            eval_map.append(i)
            
    u_out = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), cells)
        u_out[eval_map] = vals.flatten()
        
        u_ini_vals = u_initial.eval(np.array(points_on_proc), cells)
        u_ini_out = np.full((nx_out * ny_out,), np.nan)
        u_ini_out[eval_map] = u_ini_vals.flatten()
        
    u_grid = u_out.reshape(ny_out, nx_out)
    u_ini_grid = u_ini_out.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "u_initial": u_ini_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": sum(nl_iters), # approx
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nl_iters
        }
    }
