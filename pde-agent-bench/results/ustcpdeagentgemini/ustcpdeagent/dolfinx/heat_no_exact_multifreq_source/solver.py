import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract grid info
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Time parameters
    t_end = 0.12
    dt = 0.005 # Refined dt for better accuracy
    num_steps = int(np.round(t_end / dt))
    
    # Mesh and function space
    mesh_res = 100
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 2))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Current and previous solutions
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_h = fem.Function(V)
    u_h.x.array[:] = 0.0
    
    # Initial Condition for output
    u_initial = np.zeros((ny_out, nx_out))
    
    # Source term
    x = ufl.SpatialCoordinate(msh)
    f = ufl.sin(5*ufl.pi*x[0])*ufl.sin(3*ufl.pi*x[1]) + 0.5*ufl.sin(9*ufl.pi*x[0])*ufl.sin(7*ufl.pi*x[1])
    
    # Variational form (Backward Euler)
    F = ufl.inner((u - u_n) / dt, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Boundary Conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Matrix and Vector Assembly
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Linear Solver Setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    ksp_type = "cg"
    pc_type = "ilu"
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    rtol = 1e-8
    solver.setTolerances(rtol=rtol)
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for i in range(num_steps):
        t += dt
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update
        u_n.x.array[:] = u_h.x.array[:]
        
    # Interpolation on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    cells = []
    points_on_proc = []
    eval_indices = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells.append(links[0])
            eval_indices.append(i)
            
    u_eval = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_eval[eval_indices] = vals.flatten()
        
    u_grid = u_eval.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

