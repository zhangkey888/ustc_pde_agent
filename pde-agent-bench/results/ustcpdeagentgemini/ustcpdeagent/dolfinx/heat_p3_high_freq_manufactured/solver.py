import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Time parameters
    t0 = 0.0
    t_end = 0.08
    dt = 0.001
    n_steps = int(np.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    
    # Grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh
    nx_mesh, ny_mesh = 64, 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
                                   [nx_mesh, ny_mesh], cell_type=mesh.CellType.triangle)
    
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    # Exact solution for initial/boundary conditions
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, ScalarType(t0))
    
    def exact_expr(t):
        return ufl.exp(-t) * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1])
    
    # Initial condition
    u_exact_expr_0 = exact_expr(t_ufl)
    expr_0 = fem.Expression(u_exact_expr_0, V.element.interpolation_points)
    u_n.interpolate(expr_0)
    
    u_init_out = np.copy(u_n.x.array) # We will evaluate later
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Source term
    f_expr = (18*ufl.pi**2 - 1.0) * ufl.exp(-t_ufl) * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1])
    
    # Weak form: backward euler
    # (u - u_n)/dt - div(grad(u)) = f
    dt_c = fem.Constant(domain, ScalarType(dt))
    F = ufl.inner(u - u_n, v) * ufl.dx + dt_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_c * ufl.inner(f_expr, v) * ufl.dx
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
    solver.setTolerances(rtol=1e-9)
    
    u_h = fem.Function(V)
    
    total_iterations = 0
    t = t0
    
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        
        # Update BC
        expr_bc = fem.Expression(exact_expr(t_ufl), V.element.interpolation_points)
        u_bc.interpolate(expr_bc)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array
        
    # Interpolation onto output grid
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
    for i, pt in enumerate(points.T):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_out = np.full(XX.shape, np.nan).ravel()
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    # Evaluate initial condition for output
    u_n.x.array[:] = u_init_out
    u_init_out_arr = np.full(XX.shape, np.nan).ravel()
    if len(points_on_proc) > 0:
        vals_init = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_out_arr[eval_map] = vals_init.flatten()
        
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_out.reshape((ny_out, nx_out)),
        "u_initial": u_init_out_arr.reshape((ny_out, nx_out)),
        "solver_info": solver_info
    }
