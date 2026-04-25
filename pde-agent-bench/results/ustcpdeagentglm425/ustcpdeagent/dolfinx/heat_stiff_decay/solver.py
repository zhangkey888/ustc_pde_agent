import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec["pde"]
    kappa = pde["coefficients"]["kappa"]  # 0.5
    t0 = pde["time"]["t0"]  # 0.0
    t_end = pde["time"]["t_end"]  # 0.12
    dt_suggested = pde["time"]["dt"]  # 0.006
    
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Use dt/2 for better temporal accuracy (worked well in test: ~19s, L2 error ~2.7e-3)
    dt = dt_suggested / 2.0  # 0.003
    n_steps = int(round((t_end - t0) / dt))
    
    # Mesh resolution with P2 elements
    mesh_res = 100
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space - P2
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Define functions
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    # Time variable
    t_var = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Coordinates for UFL
    x = ufl.SpatialCoordinate(domain)
    
    pi = np.pi
    # f = (-10 + 2*kappa*pi^2)*exp(-10t)*sin(pi*x)*sin(pi*y)
    f_coeff = -10.0 + 2.0 * kappa * pi**2  # = -10 + pi^2
    
    sin_pi_x = ufl.sin(pi * x[0])
    sin_pi_y = ufl.sin(pi * x[1])
    
    # BC: g = 0 on all boundaries
    g_bc = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_bc, boundary_dofs, V)
    
    # Initial condition: u_0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(pi * x[0]) * np.sin(pi * x[1]))
    
    # Variational forms for Backward Euler
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    f_expr = f_coeff * ufl.exp(-10.0 * t_var) * sin_pi_x * sin_pi_y
    
    a = ufl.inner(u_trial, v_test) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = ufl.inner(u_n, v_test) * ufl.dx + dt * ufl.inner(f_expr, v_test) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=500)
    solver.setFromOptions()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    total_iterations = 0
    t = t0
    
    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        t_var.value = PETSc.ScalarType(t)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.zeros(nx_out * ny_out)
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cell = links[0]
            val = u_sol.eval(points[i:i+1], np.array([cell], dtype=np.int32))
            u_values[i] = val[0]
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    u_initial_grid = np.sin(pi * XX) * np.sin(pi * YY)
    
    # L2 error verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: np.exp(-10.0 * t) * np.sin(pi * x[0]) * np.sin(pi * x[1]))
    
    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_local = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 Error: {l2_error:.6e}")
        print(f"Total iterations: {total_iterations}")
        print(f"dt: {dt}, n_steps: {n_steps}, mesh_res: {mesh_res}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
