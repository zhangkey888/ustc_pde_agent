import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Extract params
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    t0 = 0.0
    t_end = 0.1
    dt = 0.01
    n_steps = int(np.round((t_end - t0) / dt))
    
    mesh_res = 128
    degree = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(msh, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    dt_const = fem.Constant(msh, PETSc.ScalarType(dt))
    
    # 2. Manufactured solution & source
    u_exact = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    # Exact time derivative
    u_t_exact = -u_exact
    
    # Heat equation: u_t - div(kappa*grad(u)) = f
    kappa = 1.0 + 0.5 * ufl.sin(6*ufl.pi*x[0])
    f_exact = u_t_exact - ufl.div(kappa * ufl.grad(u_exact))
    
    # 3. Setup Trial/Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    un = fem.Function(V)
    
    # Initial Condition
    u_init_expr = fem.Expression(u_exact, V.element.interpolation_points)
    un.interpolate(u_init_expr)
    
    # Keep initial condition for output
    u_initial = np.copy(un.x.array)
    
    # 4. Weak form (Backward Euler)
    # (u - un)/dt - div(kappa*grad(u)) = f  =>  u*v + dt*kappa*grad(u)*grad(v) = (un + dt*f)*v
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (un + dt_const * f_exact) * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # 5. Boundary conditions
    fdim = msh.topology.dim - 1
    # Only boundary points
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_pts: np.ones(x_pts.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Assembly & Solver
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-9)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = un.x.array[:]
    
    total_iterations = 0
    
    # 7. Time loop
    for n in range(n_steps):
        t_const.value += dt
        
        # Update BC function
        u_bc.interpolate(u_init_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply BCs to RHS
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        un.x.array[:] = u_sol.x.array[:]
    
    # 8. Output extraction
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((len(pts),), np.nan)
    u_init_vals = np.full((len(pts),), np.nan)
    
    un_temp = fem.Function(V)
    un_temp.x.array[:] = u_initial
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        
        init_vals = un_temp.eval(pts_arr, cells_arr)
        u_init_vals[eval_map] = init_vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    u_init_grid = u_init_vals.reshape((ny_out, nx_out))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-9,
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
                "nx": 64, "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    print("Wall time:", time.time() - t0)
    print("Solver info:", res["solver_info"])
