import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    # 1. Extract case specifications
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx = grid_spec.get("nx", 50)
    ny = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # 2. Solver configurations
    mesh_res = 128
    degree = 2
    t0 = 0.0
    t_end = 0.1
    dt_target = 0.01
    
    n_steps = max(1, int(math.ceil((t_end - t0) / dt_target)))
    dt = (t_end - t0) / n_steps
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    # 3. Create Mesh and Function Space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Symbolic forms for exact solution and coefficients
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y)
    u_exact = ufl.exp(-t_ufl) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    # kappa = 1 + 30*exp(-150*((x-0.35)^2 + (y-0.65)^2))
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # Source term f = ∂u/∂t - ∇·(κ ∇u)
    u_t = -ufl.exp(-t_ufl) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    flux = kappa * ufl.grad(u_exact)
    div_flux = ufl.div(flux)
    f_expr = u_t - div_flux
    
    # 5. Weak Form definition (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)
    un = fem.Function(V)
    
    def exact_solution_numpy(x_pts, t_val):
        return np.exp(-t_val) * np.sin(np.pi * x_pts[0]) * np.sin(2.0 * np.pi * x_pts[1])
    
    # Initialize un
    un.interpolate(lambda x_pts: exact_solution_numpy(x_pts, t0))
    uh.x.array[:] = un.x.array
    u_initial = un.x.array.copy()
    
    a = (u * v / dt_ufl) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (un * v / dt_ufl) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # 6. Boundary Conditions Setup
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_pts: np.full(x_pts.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    # Assemble left hand side matrix (time invariant)
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Initialize solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # 7. Time Stepping Loop
    total_iterations = 0
    t = t0
    
    for _ in range(n_steps):
        t += dt
        t_ufl.value = t
        
        # Update BC for current time
        u_bc.interpolate(lambda x_pts: exact_solution_numpy(x_pts, t))
        bcs = [fem.dirichletbc(u_bc, boundary_dofs)]
        
        # Assemble right hand side
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve system
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        un.x.array[:] = uh.x.array
        
    # 8. Evaluation on target grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
            
    # Interpolate final solution
    u_values = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    # Interpolate initial solution (optional tracking)
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u_initial
    u_init_values = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
        
    u_initial_grid = u_init_values.reshape((ny, nx))
    
    # 9. Pack results
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
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
        "u_initial": u_initial_grid
    }
