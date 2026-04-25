import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid config
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    
    # Solver config
    mesh_res = 128
    degree = 2
    dt = 0.01
    t_end = 0.2
    kappa = 0.8
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-8
    
    n_steps = int(round(t_end / dt))
    
    # Mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # BC
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs_bc)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u0_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u0_expr_eval = fem.Expression(u0_expr, V.element.interpolation_points())
    u_n.interpolate(u0_expr_eval)
    
    # Source
    f_expr = ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Weak form
    a = u * v * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt * f_expr * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Linear solver
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    
    # For output initial
    u_initial_arr = sample_on_grid(u_n, domain, bbox, nx_out, ny_out)
    
    total_iters = 0
    # Time loop
    t = 0.0
    for i in range(n_steps):
        t += dt
        
        with b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array
        
    u_arr = sample_on_grid(u_sol, domain, bbox, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_arr,
        "solver_info": solver_info,
        "u_initial": u_initial_arr
    }

def sample_on_grid(u_func, domain, bbox, nx, ny):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    return u_values.reshape(ny, nx)

