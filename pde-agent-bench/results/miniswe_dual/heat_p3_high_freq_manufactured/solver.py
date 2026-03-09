import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.08))
    dt_suggested = float(time_params.get("dt", 0.008))
    scheme = time_params.get("scheme", "backward_euler")
    
    domain_spec = case_spec.get("domain", {})
    nx_out = int(domain_spec.get("nx", 50))
    ny_out = int(domain_spec.get("ny", 50))
    
    element_degree = 2
    
    resolutions = [48, 72, 96]
    
    prev_norm = None
    final_result = None
    
    for N in resolutions:
        result = _solve_at_resolution(
            N, element_degree, kappa_val, t_end, dt_suggested, scheme, nx_out, ny_out
        )
        
        current_norm = np.linalg.norm(result["u"])
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.005:
                final_result = result
                final_result["solver_info"]["mesh_resolution"] = N
                return final_result
        
        prev_norm = current_norm
        final_result = result
        final_result["solver_info"]["mesh_resolution"] = N
    
    return final_result


def _solve_at_resolution(N, degree, kappa_val, t_end, dt, scheme, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_expr = (18.0 * ufl.pi**2 - 1.0) * ufl.exp(-t_const) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    t_const.value = 0.0
    u_init_expr_compiled = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_init_expr_compiled)
    
    u_initial_grid = _evaluate_on_grid(domain, V, u_n, nx_out, ny_out)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u * v / dt_const + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const.value = actual_dt
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    bc_expr_compiled = fem.Expression(u_exact_expr, V.element.interpolation_points)
    
    t = 0.0
    for step in range(n_steps):
        t += actual_dt
        t_const.value = t
        
        u_bc.interpolate(bc_expr_compiled)
        
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    u_grid = _evaluate_on_grid(domain, V, u_h, nx_out, ny_out)
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def _evaluate_on_grid(domain, V, u_func, nx, ny):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))
