import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion (Allen-Cahn) with manufactured solution."""
    
    start_time = time.time()
    
    pde_info = case_spec.get("pde", {})
    time_info = pde_info.get("time", {})
    output_info = case_spec.get("output", {})
    grid_info = output_info.get("grid", {})
    
    t0 = time_info.get("t0", 0.0)
    t_end = time_info.get("t_end", 0.15)
    
    nx_out = grid_info.get("nx", 50)
    ny_out = grid_info.get("ny", 50)
    bbox = grid_info.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    mesh_resolution = 64
    element_degree = 2
    dt = 0.005
    
    params = pde_info.get("parameters", {})
    epsilon = params.get("epsilon", None)
    if epsilon is None:
        epsilon = params.get("eps", 0.01)
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    # Use a fem.Function for the source term, interpolated at each time step
    f_h = fem.Function(V)
    
    # Exact solution and source as Python callables
    # u_exact(x,y,t) = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    # du/dt = -u_exact
    # lap(u) = -2*pi^2 * u_exact
    # R(u) = u^3 - u
    # f = du/dt - eps*lap(u) + R(u)
    #   = -u + eps*2*pi^2*u + u^3 - u
    #   = u*(2*eps*pi^2 - 2) + u^3
    
    _pi = np.pi
    _eps = epsilon
    
    def u_exact_func(t_val):
        def _f(xx):
            return np.exp(-t_val) * 0.3 * np.sin(_pi * xx[0]) * np.sin(_pi * xx[1])
        return _f
    
    def f_source_func(t_val):
        def _f(xx):
            u = np.exp(-t_val) * 0.3 * np.sin(_pi * xx[0]) * np.sin(_pi * xx[1])
            return u * (2.0 * _eps * _pi**2 - 2.0) + u**3
        return _f
    
    # Functions
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Initial condition
    u_n.interpolate(u_exact_func(t0))
    u_h.x.array[:] = u_n.x.array[:]
    u_h.x.scatter_forward()
    
    # Residual: (u_h - u_n)/dt + eps*grad(u_h)·grad(v) + R(u_h) - f = 0
    R_uh = u_h**3 - u_h
    F_ufl = (
        ufl.inner((u_h - u_n) / dt_const, v) * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + ufl.inner(R_uh, v) * ufl.dx
        - ufl.inner(f_h, v) * ufl.dx
    )
    J_ufl = ufl.derivative(F_ufl, u_h)
    
    F_compiled = fem.form(F_ufl)
    J_compiled = fem.form(J_ufl)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    du = fem.Function(V)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Time stepping
    t = t0
    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    newton_atol = 1e-8
    newton_max_it = 20
    
    for step in range(n_steps):
        t += dt
        
        # Update source term and BC
        f_h.interpolate(f_source_func(t))
        u_bc.interpolate(u_exact_func(t))
        
        # Initial guess
        u_h.x.array[:] = u_n.x.array[:]
        u_h.x.scatter_forward()
        
        newton_its = 0
        for newton_it in range(newton_max_it):
            A = petsc.assemble_matrix(J_compiled, bcs=[bc])
            A.assemble()
            
            b = petsc.assemble_vector(F_compiled)
            petsc.apply_lifting(b, [J_compiled], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc], u_h.x.petsc_vec)
            
            rnorm = b.norm()
            
            if rnorm < newton_atol:
                A.destroy()
                b.destroy()
                newton_its = newton_it
                break
            
            ksp.setOperators(A)
            du.x.petsc_vec.set(0.0)
            ksp.solve(b, du.x.petsc_vec)
            du.x.scatter_forward()
            total_linear_iterations += 1
            
            u_h.x.array[:] -= du.x.array[:]
            u_h.x.scatter_forward()
            
            A.destroy()
            b.destroy()
            
            newton_its = newton_it + 1
        
        nonlinear_iterations.append(newton_its)
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()
    
    ksp.destroy()
    
    # Output sampling
    u_init = fem.Function(V)
    u_init.interpolate(u_exact_func(t0))
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
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
    
    pts_arr = np.array(points_on_proc) if points_on_proc else np.empty((0, 3))
    cells_arr = np.array(cells_on_proc, dtype=np.int32) if cells_on_proc else np.empty(0, dtype=np.int32)
    
    def eval_func(func):
        vals = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            v = func.eval(pts_arr, cells_arr)
            vals[eval_map] = v.flatten()
        return vals.reshape(ny_out, nx_out)
    
    u_grid = eval_func(u_h)
    u_init_grid = eval_func(u_init)
    
    # Error
    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(u_exact_func(t_end))
    u_exact_grid = eval_func(u_exact_final)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact_grid)**2))
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.2f}s, L2 err: {error:.6e}, mesh={mesh_resolution}, deg={element_degree}, dt={dt}, steps={n_steps}", flush=True)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
