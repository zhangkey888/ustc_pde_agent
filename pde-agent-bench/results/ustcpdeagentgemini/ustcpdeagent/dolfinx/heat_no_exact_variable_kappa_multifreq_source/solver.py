import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid specification
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 128)
    ny_out = grid_spec.get("ny", 128)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Parameters
    nx, ny = 128, 128
    degree = 2
    t0 = 0.0
    t_end = 0.1
    dt = 0.005  # refined time step for better accuracy
    num_steps = int(round((t_end - t0) / dt))
    
    # Mesh
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [nx, ny], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Interpolate initial condition
    def initial_condition(x):
        return np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
    u_n.interpolate(initial_condition)
    
    # Extract initial array for tracking
    # Setup probe points
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
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
            
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)
    
    u_initial_vals = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial = u_initial_vals.reshape((ny_out, nx_out))
    
    # Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficients
    kappa_expr = 1.0 + 0.6 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    f_expr = ufl.sin(4*ufl.pi*x[0])*ufl.sin(3*ufl.pi*x[1]) + 0.3*ufl.sin(10*ufl.pi*x[0])*ufl.sin(9*ufl.pi*x[1])
    
    # Backward Euler
    F = ufl.inner(u - u_n, v) * ufl.dx + dt * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx - dt * ufl.inner(f_expr, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    total_iterations = 0
    
    # Time stepping
    t = t0
    for i in range(num_steps):
        t += dt
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array
        
    # Evaluate at output grid
    u_vals = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc, cells_on_proc)
        u_vals[eval_map] = vals.flatten()
        
    u_out = u_vals.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_out,
        "u_initial": u_initial,
        "solver_info": solver_info
    }
