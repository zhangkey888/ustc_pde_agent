import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # 1. Setup parameters
    nx_mesh = 64
    ny_mesh = 64
    deg = 2
    
    eps = 0.05
    beta_vec = [2.0, 1.0]
    
    t0 = 0.0
    t_end = 0.2
    dt = 0.01  # Smaller than suggested 0.02 for better accuracy
    num_steps = int(np.round((t_end - t0) / dt))
    
    # Grid output parameters
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # 2. Mesh and Function Space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    # 3. Time-dependent exact solution expression
    t_sym = fem.Constant(domain, PETSc.ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)
    
    # u = exp(-2*t)*sin(pi*x)*sin(pi*y)
    u_exact = ufl.exp(-2.0 * t_sym) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = u_t - eps*Laplacian(u) + beta . grad(u)
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    u_t_exact = -2.0 * u_exact
    grad_u = ufl.grad(u_exact)
    lap_u = ufl.div(grad_u)
    f_expr = u_t_exact - eps * lap_u + ufl.dot(beta, grad_u)
    
    # 4. Trial and Test Functions, and functions for time stepping
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    # Initial condition
    expr_u0 = fem.Expression(u_exact, V.element.interpolation_points)
    u_n.interpolate(expr_u0)
    
    # Capture initial condition for output
    u_initial = u_n.copy()
    
    # 5. Weak form (Backward Euler)
    # (u - u_n)/dt - eps * div(grad(u)) + beta . grad(u) = f
    # (u, v) + dt * eps * (grad(u), grad(v)) + dt * (beta . grad(u), v) = (u_n, v) + dt * (f, v)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = ufl.inner(u, v) * ufl.dx \
        + dt_const * eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + dt_const * ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        
    L = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(f_expr, v) * ufl.dx
    
    # 6. Boundary Conditions
    # All Dirichlet based on exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(expr_u0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Linear Solver Setup
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t = t0
    for i in range(num_steps):
        t += dt
        t_sym.value = t
        
        # Update BC
        u_bc.interpolate(expr_u0)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array
        
    # 8. Output interpolation
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
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
            
    u_grid = np.full(points.shape[1], np.nan)
    u_init_grid = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
        
        vals_init = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_grid[eval_map] = vals_init.flatten()
        
    u_grid = u_grid.reshape((ny_out, nx_out))
    u_init_grid = u_init_grid.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": deg,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50, "ny": 50,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    res = solve(case_spec)
    print("Solve completed.")
    print("Iterations:", res["solver_info"]["iterations"])
