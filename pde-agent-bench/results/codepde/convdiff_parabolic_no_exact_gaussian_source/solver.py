import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Diffusion coefficient
    epsilon = float(pde.get("epsilon", 0.02))
    
    # Convection velocity
    beta_vec = pde.get("beta", [6.0, 2.0])
    beta_x = float(beta_vec[0])
    beta_y = float(beta_vec[1])
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Source term string
    source_str = pde.get("source", "exp(-200*((x-0.3)**2 + (y-0.7)**2))*exp(-t)")
    
    # Initial condition
    ic_str = pde.get("initial_condition", "0.0")
    
    # Boundary condition
    bc_val_str = pde.get("boundary_condition", "0.0")
    
    # 2. Create mesh - use finer mesh for high Peclet number
    N = 100
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P1
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_val = dt_suggested
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Convection velocity
    beta = ufl.as_vector([fem.Constant(domain, PETSc.ScalarType(beta_x)),
                           fem.Constant(domain, PETSc.ScalarType(beta_y))])
    
    # Diffusion constant
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Source term: f = exp(-200*((x-0.3)**2 + (y-0.7)**2))*exp(-t)
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) * ufl.exp(-t_const)
    
    # 5. Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    # Set initial condition
    u_n.interpolate(lambda x: np.zeros(x.shape[1]))
    
    # 6. Boundary conditions - homogeneous Dirichlet
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 7. SUPG Stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    
    # SUPG stabilization parameter
    tau = h / (2.0 * beta_mag) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 
                                                     1.0 - 1.0/Pe_cell, 
                                                     0.0))
    
    # 8. Variational form - Backward Euler with SUPG
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form: (u - u_n)/dt * v + eps*grad(u).grad(v) + beta.grad(u)*v = f*v
    # Plus SUPG terms
    
    # Standard Galerkin terms
    a_std = (u * v / dt_c * ufl.dx
             + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    
    L_std = (u_n * v / dt_c * ufl.dx
             + f_expr * v * ufl.dx)
    
    # SUPG stabilization: add tau * (beta . grad(v)) * residual
    # For backward Euler, the strong residual is:
    # R = (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # So R ≈ (u - u_n)/dt + beta.grad(u) - f
    
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = (u / dt_c * supg_test * ufl.dx
              + ufl.dot(beta, ufl.grad(u)) * supg_test * ufl.dx)
    
    L_supg = (u_n / dt_c * supg_test * ufl.dx
              + f_expr * supg_test * ufl.dx)
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # 9. Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # 10. Assemble matrix (LHS doesn't change if coefficients are constant in time,
    # but tau depends on mesh only, eps/beta are constant, dt is constant)
    # Actually the bilinear form doesn't depend on time, so we assemble once
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)  # we'll use petsc vector
    b_vec = petsc.create_vector(L_form)
    
    # 11. Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # 12. Time stepping
    n_steps = int(np.ceil(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    dt_c.value = dt_val
    
    # Need to reassemble A since dt changed
    A.zeroEntries()
    petsc.assemble_matrix(A, a_form, bcs=[bc])
    A.assemble()
    solver.setOperators(A)
    
    total_iterations = 0
    t_current = 0.0
    
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current  # update time for source term
        
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
    
    # 13. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition (which is zero)
    u_initial = np.zeros((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "stabilization": "SUPG",
        }
    }