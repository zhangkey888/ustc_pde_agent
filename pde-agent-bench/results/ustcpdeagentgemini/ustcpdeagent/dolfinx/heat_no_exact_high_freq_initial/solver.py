import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Extract parameters
    t_end = 0.005
    dt = 0.001
    kappa = 1.0
    
    mesh_res = 128
    degree = 2
    
    # Output grid parameters
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    x_ufl = ufl.SpatialCoordinate(domain)
    u0_expr = ufl.sin(6 * ufl.pi * x_ufl[0]) * ufl.sin(6 * ufl.pi * x_ufl[1])
    u0_expr_eval = fem.Expression(u0_expr, V.element.interpolation_points)
    u_n.interpolate(u0_expr_eval)
    
    u_init_vals = u_n.x.array.copy()
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bdofs)
    
    # Variational Problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = u * v * ufl.dx + fem.Constant(domain, PETSc.ScalarType(dt * kappa)) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-9)
    
    u_sol = fem.Function(V)
    
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iters = 0
    
    for i in range(n_steps):
        t += dt
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iters += solver.getIterationNumber()
        
        # Update
        u_n.x.array[:] = u_sol.x.array
        u_n.x.scatter_forward()
        
    # Evaluate on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())])
    
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
            
    u_grid_flat = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    u_grid = u_grid_flat.reshape((ny_out, nx_out))
    
    # Also evaluate initial condition for output
    u_n.x.array[:] = u_init_vals
    u_init_grid_flat = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_grid_flat[eval_map] = vals.flatten()
    u_init_grid = u_init_grid_flat.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
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
                "nx": 100,
                "ny": 100,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    res = solve(case_spec)
    print("Done")
