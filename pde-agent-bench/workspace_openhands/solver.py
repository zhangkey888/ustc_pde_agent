import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    kappa = 10.0
    t_end = 0.05
    dt = 0.005
    n_steps = int(round(t_end / dt))
    
    mesh_resolution = 64
    degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    
    # u0 = sin(pi*x)*sin(pi*y)
    u0_expr = fem.Expression(ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]), V.element.interpolation_points)
    u_n.interpolate(u0_expr)
    
    def sample_on_grid(u_func, nx, ny):
        x_coords = np.linspace(0, 1, nx)
        y_coords = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
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
                
        u_values = np.full((points.shape[1],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        return u_values.reshape((nx, ny))

    nx_eval, ny_eval = 50, 50
    u_initial_grid = sample_on_grid(u_n, nx_eval, ny_eval)
    
    # Boundary condition (u=0 on boundary)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    t = 0.0
    f_func = fem.Function(V)
    
    def f_eval(x_pts, current_t):
        val = (2.0 * kappa * np.pi**2 - 1.0) * np.exp(-current_t) * np.sin(np.pi * x_pts[0]) * np.sin(np.pi * x_pts[1])
        return val
    
    # Variational problem (Backward Euler)
    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    solver.setTolerances(rtol=1e-9)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    for i in range(n_steps):
        t += dt
        
        # Update source term
        f_func.interpolate(lambda x_pts: f_eval(x_pts, t))
        
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
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array[:]
        
    u_final_grid = sample_on_grid(u_sol, nx_eval, ny_eval)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "rtol": 1e-9,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_final_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }
