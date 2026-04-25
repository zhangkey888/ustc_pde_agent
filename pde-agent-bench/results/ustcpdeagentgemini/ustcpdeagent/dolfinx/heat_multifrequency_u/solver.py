import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc as petsc
import ufl

def solve(case_spec: dict) -> dict:
    # 1. Parse parameters
    # Domain is unit square [0,1] x [0,1]
    # Time parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025  # Using smaller dt for higher accuracy
    
    # Grid output parameters
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]

    # Agent parameters
    mesh_res = 128
    element_deg = 2
    
    # 2. Setup Mesh and Function Space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_deg))
    
    # 3. Define Exact Solution and Source Term Symbolically
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    def u_exact_ufl(time_c):
        return ufl.exp(-time_c) * (ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) + 0.2*ufl.sin(6*ufl.pi*x[0])*ufl.sin(6*ufl.pi*x[1]))
    
    # Source term: f = du/dt - div(grad(u))
    # du/dt = -u_exact (since exp(-t) derivative is -exp(-t))
    f_expr = -u_exact_ufl(t) - ufl.div(ufl.grad(u_exact_ufl(t)))
    
    # 4. Trial and Test Functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # 5. Functions for time stepping
    u_n = fem.Function(V)
    
    # Initial Condition
    u_init_expr = fem.Expression(u_exact_ufl(t), V.element.interpolation_points())
    u_n.interpolate(u_init_expr)
    
    # For output analysis
    u_initial_array = None

    # 6. Variational Problem (Backward Euler)
    # (u - u_n)/dt - div(grad(u)) = f
    # u*v + dt*inner(grad(u), grad(v)) = u_n*v + dt*f*v
    F = u*v*ufl.dx + dt*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - u_n*v*ufl.dx - dt*f_expr*v*ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # 7. Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_init_expr) # Initially at t0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 8. Setup Solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # 9. Interpolation Utility
    def probe_grid(u_func, nx, ny, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
        
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(pts.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(pts.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
                
        u_vals = np.full(pts.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_vals[eval_map] = vals.flatten()
            
        return u_vals.reshape((ny, nx))
        
    u_initial_array = probe_grid(u_n, nx_out, ny_out, bbox)

    # 10. Time Stepping Loop
    t_current = t0
    n_steps = 0
    total_linear_iterations = 0
    
    while t_current < t_end - 1e-8:
        t_current += dt_val
        t.value = t_current
        
        # Update BC
        u_bc_expr = fem.Expression(u_exact_ufl(t), V.element.interpolation_points())
        u_bc.interpolate(u_bc_expr)
        
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
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array[:]
        n_steps += 1
        total_linear_iterations += solver.getIterationNumber()
        
    # 11. Final output
    u_out = probe_grid(u_sol, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_deg,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": total_linear_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_out,
        "u_initial": u_initial_array,
        "solver_info": solver_info
    }

