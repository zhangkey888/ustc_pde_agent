import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    eps = float(pde.get("epsilon", pde.get("eps", 1e-2)))
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])
    dt = float(pde["time"]["dt"])
    
    output = case_spec["output"]
    nx_out = output["grid"]["nx"]
    ny_out = output["grid"]["ny"]
    bbox = output["grid"]["bbox"]
    xmin, xmax, ymin, ymax = [float(b) for b in bbox]
    
    mesh_res = 48
    elem_deg = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    n_steps = int(round((t_end - t0) / dt))
    time_scheme = "backward_euler"
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    u_ex = ufl.exp(-t_const) * 0.25 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    du_dt = -ufl.exp(-t_const) * 0.25 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    lap_u_ex = ufl.div(ufl.grad(u_ex))
    f_source = du_dt - eps * lap_u_ex + u_ex**3 - u_ex
    
    u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points)
    
    u = fem.Function(V)
    u_prev = fem.Function(V)
    
    t_const.value = PETSc.ScalarType(t0)
    u.interpolate(u_ex_expr)
    u_prev.interpolate(u_ex_expr)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_ex_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    v = ufl.TestFunction(V)
    F = (u - u_prev) / dt * v * ufl.dx \
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u**3 - u) * v * ufl.dx \
        - f_source * v * ufl.dx
    
    J = ufl.derivative(F, u)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 20,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J,
                                      petsc_options_prefix="ac_",
                                      petsc_options=petsc_options)
    
    total_ksp_iters = 0
    nonlinear_iters = []
    
    for step in range(n_steps):
        current_t = t0 + (step + 1) * dt
        t_const.value = PETSc.ScalarType(current_t)
        u_bc_func.interpolate(u_ex_expr)
        u_prev.x.array[:] = u.x.array[:]
        
        problem.solve()
        u.x.scatter_forward()
        
        snes = problem._snes
        nl_it = int(snes.getIterationNumber())
        nonlinear_iters.append(nl_it)
        ksp = snes.ksp
        total_ksp_iters += ksp.getIterationNumber()
    
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    t_const.value = PETSc.ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_ex_expr)
    
    error_form = fem.form(ufl.inner(u - u_exact_func, u - u_exact_func) * ufl.dx)
    l2_error = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"L2 error at t={t_end}: {l2_error:.6e}")
    
    t_const.value = PETSc.ScalarType(t0)
    u_ic = fem.Function(V)
    u_ic.interpolate(u_ex_expr)
    
    ic_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_ic = u_ic.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        ic_values[eval_map] = vals_ic.flatten()
    u_initial_grid = ic_values.reshape(ny_out, nx_out)
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": total_ksp_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iters,
        },
        "u_initial": u_initial_grid,
    }
    
    return result
