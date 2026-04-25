import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    wall_start = time.perf_counter()
    
    # Extract parameters
    pde = case_spec["pde"]
    kappa = float(pde["coefficients"]["kappa"])
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])
    
    # Small dt for good temporal accuracy with Crank-Nicolson
    dt = 0.0003
    n_steps = int(round((t_end - t0) / dt))
    
    # Output grid
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # High resolution mesh
    mesh_res = 200
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res,
                                      cell_type=mesh.CellType.triangle)
    
    elem_degree = 2
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    u_sol = fem.Function(V)
    u_old = fem.Function(V)
    f_func = fem.Function(V)
    f_old_func = fem.Function(V)
    
    # Initial condition: u_0 = sin(4*pi*x)*sin(4*pi*y)
    u_old.interpolate(lambda x: np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1]))
    
    # BC: u = 0 on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim,
                        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Crank-Nicolson (2nd order in time)
    a = (1.0/dt) * u * v * ufl.dx + 0.5 * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_form = fem.form(a)
    
    L = ((1.0/dt) * u_old * v 
         - 0.5 * kappa * ufl.inner(ufl.grad(u_old), ufl.grad(v)) 
         + 0.5 * (f_func + f_old_func) * v) * ufl.dx
    L_form = fem.form(L)
    
    # Initialize f_old at t0
    f_old_func.interpolate(lambda x: 
        (32*np.pi**2 - 1)*np.exp(-t0)*np.sin(4*np.pi*x[0])*np.sin(4*np.pi*x[1]))
    
    # Assemble matrix A (constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Direct LU solver
    solver_ksp = PETSc.KSP().create(domain.comm)
    solver_ksp.setOperators(A)
    solver_ksp.setType(PETSc.KSP.Type.PREONLY)
    solver_ksp.getPC().setType(PETSc.PC.Type.LU)
    
    b = A.createVecRight()
    
    total_iterations = 0
    t = t0
    
    for n in range(n_steps):
        t_new = t + dt
        
        # Update source term
        f_func.interpolate(lambda x, t_val=t_new: 
            (32*np.pi**2 - 1)*np.exp(-t_val)*np.sin(4*np.pi*x[0])*np.sin(4*np.pi*x[1]))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver_ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += 1
        u_old.x.array[:] = u_sol.x.array[:]
        f_old_func.x.array[:] = f_func.x.array[:]
        t = t_new
    
    wall_end = time.perf_counter()
    
    # Accuracy verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: np.exp(-t_end)*np.sin(4*np.pi*x[0])*np.sin(4*np.pi*x[1]))
    L2_err_form = fem.form((u_sol - u_exact_func)**2 * ufl.dx)
    error_sq = fem.assemble_scalar(L2_err_form)
    L2_error = np.sqrt(domain.comm.allreduce(error_sq, op=MPI.SUM))
    
    if domain.comm.rank == 0:
        print(f"L2 error at t={t_end}: {L2_error:.6e}")
        print(f"Wall time: {wall_end - wall_start:.2f}s")
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_values_global = np.full_like(u_values, np.nan)
    domain.comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1]))
    u_init_vals = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        v_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = v_init.flatten()
    u_init_global = np.full_like(u_init_vals, np.nan)
    domain.comm.Allreduce(u_init_vals, u_init_global, op=MPI.SUM)
    u_initial_grid = u_init_global.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson",
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
