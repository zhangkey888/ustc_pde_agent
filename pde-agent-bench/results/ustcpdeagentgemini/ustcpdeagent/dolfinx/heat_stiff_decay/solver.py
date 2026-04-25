import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # Time parameters
    t0 = 0.0
    t_end = 0.12
    dt = 0.006
    n_steps = int(np.round((t_end - t0) / dt))
    
    # Mesh parameters
    nx_mesh = 64
    ny_mesh = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution expression
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    u_exact_ufl = ufl.exp(-10.0 * t_ufl) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Output grid setup
    out_nx = case_spec["output"]["grid"]["nx"]
    out_ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], out_nx)
    ys = np.linspace(bbox[2], bbox[3], out_ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
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
            
    u_init_vals = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals.flatten()
    u_initial = u_init_vals.reshape((out_ny, out_nx))
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa = 0.5
    f = (ufl.pi**2 - 10.0) * u_exact_ufl
    
    # Backward Euler
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    F = ufl.inner((u - u_n) / dt_ufl, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b_vec = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("ilu")
    
    u_h = fem.Function(V)
    
    total_iterations = 0
    t = t0
    
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        
        # Update BC and forcing
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array
        
    u_out_vals = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out_vals[eval_map] = vals.flatten()
    u_grid = u_out_vals.reshape((out_ny, out_nx))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-5,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial
    }
