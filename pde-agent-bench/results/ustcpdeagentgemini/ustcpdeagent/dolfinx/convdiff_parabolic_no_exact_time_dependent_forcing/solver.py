import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc as petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh
    nx = 64
    ny = 64
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Parameters
    epsilon = 0.02
    beta = ufl.as_vector([6.0, 3.0])
    
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.01  # Smaller than suggested for better accuracy
    n_steps = int(round((t_end - t0) / dt_val))
    
    # Boundary Conditions (u = 0 on all walls)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    # Probe points
    out_nx = case_spec["output"]["grid"]["nx"]
    out_ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], out_nx)
    ys = np.linspace(bbox[2], bbox[3], out_ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(out_nx * out_ny)].T
    
    # Evaluate Initial Condition
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_initial_vals = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals.flatten()
    u_initial = u_initial_vals.reshape(out_ny, out_nx)
    
    # Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-150*((x[0]-0.4)**2 + (x[1]-0.6)**2)) * ufl.exp(-t)
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm)
    
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Residual
    # Strong residual for P1 elements (ignoring diffusion second derivative)
    R = (u - u_n)/dt + ufl.dot(beta, ufl.grad(u)) - f
    
    # Standard Galerkin
    F_galerkin = ((u - u_n)/dt * v + epsilon * ufl.dot(ufl.grad(u), ufl.grad(v)) 
                  + ufl.dot(beta, ufl.grad(u)) * v - f * v) * ufl.dx
    
    # SUPG term
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    F_supg = R * v_supg * ufl.dx
    
    F = F_galerkin + F_supg
    
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Setup solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    total_iters = 0
    
    # Time loop
    current_t = t0
    for n in range(n_steps):
        current_t += dt_val
        t.value = current_t
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
        
    # Evaluate Solution
    u_vals = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(out_ny, out_nx)
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iters,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial
    }
