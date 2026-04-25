import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    coeff = pde["coefficients"]
    kappa = float(coeff["kappa"])
    
    time_info = pde["time"]
    t0 = float(time_info["t0"])
    t_end = float(time_info["t_end"])
    dt_suggested = float(time_info["dt"])
    
    output_info = case_spec["output"]
    grid = output_info["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    mesh_res = 96
    elem_degree = 2
    dt = dt_suggested / 2
    time_scheme = "backward_euler"
    rtol = 1e-9
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    u_exact_ufl = ufl.exp(-2 * t_const) * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    f_coeff_val = -2.0 + 2.0 * kappa * pi**2
    f_ufl = f_coeff_val * ufl.exp(-2 * t_const) * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    u_n = fem.Function(V)
    u1 = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u1 * v / dt_const + kappa * ufl.inner(ufl.grad(u1), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    u_n.interpolate(
        fem.Expression(ufl.cos(pi * x[0]) * ufl.cos(pi * x[1]), V.element.interpolation_points)
    )
    
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
    ksp.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        t_const.value = ScalarType(t)
        u_bc_func.interpolate(u_bc_expr)
        
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
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=MPI.SUM)
        nm = np.isnan(u_values).astype(np.float64)
        nmg = np.zeros_like(nm)
        comm.Allreduce(nm, nmg, op=MPI.SUM)
        u_global[nmg > 0.5] = np.nan
        u_values = u_global
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(
        fem.Expression(ufl.cos(pi * x[0]) * ufl.cos(pi * x[1]), V.element.interpolation_points)
    )
    u_init_values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vi = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vi.flatten()
    
    if comm.size > 1:
        ui_global = np.zeros_like(u_init_values)
        comm.Allreduce(u_init_values, ui_global, op=MPI.SUM)
        nmi = np.isnan(u_init_values).astype(np.float64)
        nmig = np.zeros_like(nmi)
        comm.Allreduce(nmi, nmig, op=MPI.SUM)
        ui_global[nmig > 0.5] = np.nan
        u_init_values = ui_global
    
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        },
        "u_initial": u_initial_grid,
    }
