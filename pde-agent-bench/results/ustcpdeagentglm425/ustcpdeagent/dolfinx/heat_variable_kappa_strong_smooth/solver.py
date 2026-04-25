import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    pde = case_spec["pde"]
    time_params = pde.get("time", {})
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.01))
    
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    xmin, xmax, ymin, ymax = [float(b) for b in bbox]
    
    mesh_res = 80
    element_degree = 2
    dt_use = 0.005
    time_scheme = "backward_euler"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    
    # Time variable
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Exact solution: u = exp(-t)*sin(3*pi*x)*sin(2*pi*y)
    # Kappa: 1 + 0.8*sin(2*pi*x)*sin(2*pi*y)
    # Pre-computed source term f = du/dt - div(kappa*grad(u)):
    # f = -exp(-t)*sin(3*pi*x)*sin(2*pi*y)
    #   + exp(-t)*sin(2*pi*y)*9*pi^2*cos(3*pi*x)*2*pi*0.8*cos(2*pi*x)
    #   + exp(-t)*sin(3*pi*x)*4*pi^2*cos(2*pi*y)*2*pi*0.8*cos(2*pi*y)
    #   + exp(-t)*(1+0.8*sin(2*pi*x)*sin(2*pi*y))*(9*pi^2*sin(3*pi*x)*sin(2*pi*y)+4*pi^2*sin(3*pi*x)*sin(2*pi*y))
    # Let me use UFL but keep it simpler by building f from parts
    
    pi = ufl.pi
    et = ufl.exp(-t_const)
    s3x = ufl.sin(3*pi*x[0])
    s2y = ufl.sin(2*pi*x[1])
    c3x = ufl.cos(3*pi*x[0])
    c2y = ufl.cos(2*pi*x[1])
    s2x = ufl.sin(2*pi*x[0])
    
    kappa = 1.0 + 0.8 * s2x * s2y
    
    # du/dt = -et*s3x*s2y
    # grad(u) = et*[3*pi*c3x*s2y, 2*pi*s3x*c2y]
    # div(kappa*grad(u)):
    #   d/dx[kappa*et*3*pi*c3x*s2y] + d/dy[kappa*et*2*pi*s3x*c2y]
    # = et*s2y*3*pi*c3x * dkappa/dx + kappa*et*3*pi*(-3*pi*s3x)*s2y
    # + et*3*pi*c3x * dkappa/dy*s2y + kappa*et*2*pi*s3x*(-2*pi*c2y)*0 (wait no)
    # Let me be more careful:
    # d/dx[kappa * et * 3*pi*cos(3*pi*x)*sin(2*pi*y)]
    # = et*sin(2*pi*y)*3*pi*cos(3*pi*x) * d(kappa)/dx 
    #   + kappa * et * 3*pi * (-3*pi*sin(3*pi*x)) * sin(2*pi*y)
    # d(kappa)/dx = 0.8 * 2*pi*cos(2*pi*x)*sin(2*pi*y)
    # d(kappa)/dy = 0.8 * 2*pi*sin(2*pi*x)*cos(2*pi*y)
    
    # d/dy[kappa * et * 2*pi*sin(3*pi*x)*cos(2*pi*y)]
    # = et*sin(3*pi*x)*2*pi*cos(2*pi*y) * d(kappa)/dy
    #   + kappa * et * 2*pi*sin(3*pi*x) * (-2*pi*sin(2*pi*y))
    
    # So div(kappa*grad(u)) = 
    #   et*s2y*3*pi*c3x*0.8*2*pi*c2x*s2y    [dkappa/dx term]
    # + kappa*et*(-9*pi^2)*s3x*s2y          [d2u/dx2 term]  
    # + et*s3x*2*pi*c2y*0.8*2*pi*s2x*c2y   [dkappa/dy term]
    # + kappa*et*(-4*pi^2)*s3x*s2y          [d2u/dy2 term]
    
    # f = du/dt - div(kappa*grad(u))
    # = -et*s3x*s2y - [above]
    
    # Build f in UFL directly:
    dkappa_dx = 0.8 * 2*pi*ufl.cos(2*pi*x[0]) * s2y
    dkappa_dy = 0.8 * 2*pi * s2x * ufl.cos(2*pi*x[1])
    
    div_kappa_grad_u = (et*s2y*3*pi*c3x*dkappa_dx 
                       + kappa*et*(-9*pi**2)*s3x*s2y
                       + et*s3x*2*pi*c2y*dkappa_dy
                       + kappa*et*(-4*pi**2)*s3x*s2y)
    
    f_source = -et*s3x*s2y - div_kappa_grad_u
    
    # Exact solution UFL (for BC and initial condition)
    u_exact_ufl = et * s3x * s2y
    
    # Time stepping
    n_steps = int(round((t_end - t0) / dt_use))
    dt_actual = (t_end - t0) / n_steps
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_actual))
    
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    u_bc_func = fem.Function(V)
    
    # Initial condition
    t_const.value = PETSc.ScalarType(t0)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    u_initial_grid = _sample_on_grid(u_n, nx_out, ny_out, bbox, domain)
    
    # BCs
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Variational forms
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (ufl.inner(u_trial, v) / dt_const * ufl.dx
         + ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx)
    L = (ufl.inner(u_n, v) / dt_const * ufl.dx
         + ufl.inner(f_source, v) * ufl.dx)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")
    ksp.setTolerances(rtol=rtol, atol=1e-12)
    ksp.setFromOptions()
    
    total_iterations = 0
    
    for step in range(n_steps):
        current_time = t0 + (step + 1) * dt_actual
        t_const.value = PETSc.ScalarType(current_time)
        u_bc_func.interpolate(bc_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    u_grid = _sample_on_grid(u_sol, nx_out, ny_out, bbox, domain)
    
    # Verification
    t_const.value = PETSc.ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(bc_expr)
    diff = u_sol - u_exact_func
    error_sq = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    l2_error = np.sqrt(domain.comm.allreduce(error_sq, op=MPI.SUM))
    
    wall_time = time.time() - start_time
    print(f"[solver] L2={l2_error:.6e} time={wall_time:.2f}s iters={total_iterations} steps={n_steps}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        }
    }

def _sample_on_grid(u_func, nx, ny, bbox, domain):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    gathered = np.full_like(u_values, np.nan)
    domain.comm.Allreduce(u_values, gathered, op=MPI.SUM)
    return gathered.reshape(ny, nx)
