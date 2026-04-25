import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import math

def solve(case_spec: dict) -> dict:
    # Extract time parameters
    t0 = 0.0
    t_end = 0.06
    dt = 0.005
    n_steps = int(round((t_end - t0) / dt))
    
    # Mesh parameters
    nx_mesh = 128
    ny_mesh = 128
    degree = 2
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # PDE Parameters
    epsilon = 0.01
    beta = ufl.as_vector([12.0, 4.0])
    
    # Time variable for exact solution
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution expression
    u_ex_expr = ufl.exp(-t_ufl) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute f analytically
    u_t_ex = -u_ex_expr
    grad_u_ex = ufl.grad(u_ex_expr)
    laplace_u_ex = -17 * ufl.pi**2 * u_ex_expr
    f_expr = u_t_ex - epsilon * laplace_u_ex + ufl.dot(beta, grad_u_ex)
    
    # Define test and trial functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Current and previous solutions
    u_n = fem.Function(V)
    
    # Set initial condition
    u_n_expr = fem.Expression(u_ex_expr, V.element.interpolation_points)
    u_n.interpolate(u_n_expr)
    
    # To store initial condition for output
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(u_n_expr) # at t0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    v_norm = ufl.sqrt(ufl.dot(beta, beta))
    # Pe = v_norm * h / (2 * epsilon)
    tau = h / (2.0 * v_norm)
    
    # Residual of the strong form
    u_mid = u  # Backward Euler
    res_strong = (u - u_n) / dt_ufl - epsilon * ufl.div(ufl.grad(u_mid)) + ufl.dot(beta, ufl.grad(u_mid)) - f_expr
    
    # Galerkin formulation
    F_galerkin = (ufl.inner(u - u_n, v) / dt_ufl) * ufl.dx \
               + epsilon * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx \
               + ufl.inner(ufl.dot(beta, ufl.grad(u_mid)), v) * ufl.dx \
               - ufl.inner(f_expr, v) * ufl.dx
               
    # SUPG formulation
    F_supg = F_galerkin + tau * ufl.inner(res_strong, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a = ufl.lhs(F_supg)
    L = ufl.rhs(F_supg)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t = t0
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        
        # Update boundary condition
        u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
        
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
        
    # Sample on grid
    def sample_on_grid(u_func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, pts)
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

        u_values = np.full((pts.shape[0],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        # Optional: reduce over MPI if needed
        u_values_global = comm.allreduce(np.nan_to_num(u_values, nan=0.0), op=MPI.SUM)
        
        # Since each point is in exactly one processor (or on boundary), we need to avoid double counting, 
        # but for simplicity on 1 process, this is fine.
        return u_values.reshape((ny_out, nx_out))

    u_grid = sample_on_grid(u_sol)
    u_init_grid = sample_on_grid(u_initial)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }

