import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Extract case specs
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 128
    element_deg = 1
    dt_val = 0.005
    t0 = 0.0
    t_end = 0.1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_deg))
    
    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Exact solution
    u_ex_expr = ufl.exp(-t_const) * ufl.exp(-40 * r2)
    # Source term: f = du/dt - div(grad(u))
    # Since u_ex = exp(-t) * ..., du/dt = -u_ex
    f = -u_ex_expr - ufl.div(ufl.grad(u_ex_expr))
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points()))
    u_init = u_n.copy()
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler Form
    F = (u - u_n) / dt_const * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    
    u_h = fem.Function(V)
    
    t = t0
    total_iters = 0
    n_steps = 0
    
    while t < t_end - 1e-8:
        t += dt_val
        t_const.value = t
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points()))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
        n_steps += 1
        
    # Evaluate on target grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
    u_init_values = np.full((points.shape[1],), np.nan)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        vals_init = u_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    u_init_grid = u_init_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-5,
        "iterations": total_iters,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_init_grid
    }
