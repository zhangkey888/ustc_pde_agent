import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Parameters
    nx = 64
    ny = 64
    degree = 2
    t_end = 0.12
    dt = 0.01
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    
    comm = MPI.COMM_WORLD
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [nx, ny], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u_initial_array = u_n.x.array.copy() # Just for reference if needed
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Backward Euler: (u - u_n)/dt - kappa * Laplacian(u) = f
    # -> u/dt - kappa * Laplacian(u) = f + u_n/dt
    # -> (u, v) + dt * kappa * (grad u, grad v) = dt * (f, v) + (u_n, v)
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_const * f) * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    for i in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on grid
    out_nx = case_spec["output"]["grid"]["nx"]
    out_ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], out_nx)
    ys = np.linspace(bbox[2], bbox[3], out_ny)
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    # Same for initial condition
    u_initial = fem.Function(V)
    u_initial.x.array[:] = 0.0
    u_initial_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_values[eval_map] = vals_init.flatten()
        
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_values.reshape(out_ny, out_nx),
        "u_initial": u_initial_values.reshape(out_ny, out_nx),
        "solver_info": solver_info
    }

