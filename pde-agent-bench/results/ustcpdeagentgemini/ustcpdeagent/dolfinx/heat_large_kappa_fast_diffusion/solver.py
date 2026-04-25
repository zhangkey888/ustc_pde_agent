import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Solver parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-8
    dt = 0.004
    t_end = 0.08
    kappa = 5.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution expression
    u_ex_expr = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_initial = fem.Function(V)
    u_initial.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
    u_n.x.array[:] = u_initial.x.array
    
    # Source term
    f = (-1.0 + 25.0 * ufl.pi**2) * ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Weak form (Backward Euler)
    F = ufl.inner(u - u_n, v) * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_c * ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Setup solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    current_time = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for i in range(n_steps):
        current_time += dt
        t.value = current_time
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
        
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
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
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
    u_init_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
        init_vals = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = init_vals.flatten()
        
    u_grid = u_values.reshape(ny_out, nx_out)
    u_init_grid = u_init_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_init_grid
    }
