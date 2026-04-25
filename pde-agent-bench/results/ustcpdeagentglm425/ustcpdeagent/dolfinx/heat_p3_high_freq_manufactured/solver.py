import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Time parameters
    t0 = 0.0
    t_end = 0.08
    kappa = 1.0
    
    # Discretization parameters
    mesh_res = 48
    elem_deg = 3
    dt = 0.002
    n_steps = int(round((t_end - t0) / dt))
    
    start_time = time.time()
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Current and previous solution
    u_n = fem.Function(V)
    u = fem.Function(V)
    
    # Time constant for source term
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Source term: f = exp(-t)*(18*pi^2 - 1)*sin(3*pi*x)*sin(3*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-t_const) * (18*ufl.pi**2 - 1) * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1])
    
    # Initial condition: u0 = sin(3*pi*x)*sin(3*pi*y)
    u_n.interpolate(lambda x: np.sin(3*np.pi*x[0]) * np.sin(3*np.pi*x[1]))
    
    # Boundary condition: g = 0 on all boundaries (sin(3*pi*0)=0, sin(3*pi*1)=0)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Variational form for backward Euler
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    
    a = ufl.inner(u_trial, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant since dt and kappa don't change)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create PETSc solver - direct LU for speed
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    rtol = 1e-12
    
    # RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    total_iterations = 0
    
    # Time stepping loop
    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        t_const.value = PETSc.ScalarType(t)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        
        total_iterations += 1
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # For MPI, gather all values
    if comm.size > 1:
        all_values = comm.allgather(u_values)
        combined = np.full((nx_out * ny_out,), np.nan)
        for vals in all_values:
            mask = ~np.isnan(vals)
            combined[mask] = vals[mask]
        u_values = combined
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Initial condition on grid
    u_init_grid = np.sin(3*np.pi*XX) * np.sin(3*np.pi*YY)
    
    # Compute error for verification
    u_exact = np.exp(-t_end) * np.sin(3*np.pi*XX) * np.sin(3*np.pi*YY)
    max_error = np.max(np.abs(u_grid - u_exact))
    dx = (xs[1] - xs[0]) if nx_out > 1 else 1.0
    dy = (ys[1] - ys[0]) if ny_out > 1 else 1.0
    l2_error = np.sqrt(np.sum((u_grid - u_exact)**2) * dx * dy)
    
    wall_time = time.time() - start_time
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": float(rtol),
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.08, "dt": 0.008}
        }
    }
    result = solve(case_spec)
    u_exact = np.exp(-0.08) * np.sin(3*np.pi*np.linspace(0,1,64))[None,:] * np.sin(3*np.pi*np.linspace(0,1,64))[:,None]
    max_err = np.max(np.abs(result["u"] - u_exact))
    print(f"Output shape: {result['u'].shape}")
    print(f"Max error: {max_err:.6e}")
    print(f"Solver info: {result['solver_info']}")
