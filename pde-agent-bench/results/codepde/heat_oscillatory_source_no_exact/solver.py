import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    kappa = 0.8
    t_end = 0.12
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Try to parse from case_spec
    if "coefficients" in pde:
        kappa = float(pde["coefficients"].get("kappa", kappa))
    if "time" in pde:
        t_end = float(pde["time"].get("t_end", t_end))
        dt_suggested = float(pde["time"].get("dt", dt_suggested))
        scheme = pde["time"].get("scheme", scheme)
    
    # Use a finer dt for accuracy
    dt = 0.005
    n_steps = int(np.round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 80
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.x.array[:] = 0.0  # initial condition u0 = 0
    
    # Store initial condition for output
    # Evaluate on grid before time stepping
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    def evaluate_on_grid(u_func):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points_3d.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points_3d.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full(points_3d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)
    
    u_initial = evaluate_on_grid(u_n)
    
    # 5. Variational forms for backward Euler
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (f, v) + (u_n, v)/dt
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (f * v + u_n * v / dt_const) * ufl.dx
    
    # 6. Boundary conditions: u = g on boundary
    # g = 0 (homogeneous Dirichlet, since initial condition is 0 and no BC specified otherwise)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 7. Assemble and solve using manual assembly for time loop efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    u_sol = fem.Function(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_grid = evaluate_on_grid(u_sol)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }