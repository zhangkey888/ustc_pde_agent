import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out, ny_out = grid_spec["nx"], grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Solver parameters
    mesh_resolution = 64
    degree = 2
    t0 = 0.0
    t_end = 0.1
    dt = 0.005  # refined time step for higher accuracy
    
    # Mesh
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [mesh_resolution, mesh_resolution], 
        mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution expression
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    u_exact_ufl = ufl.exp(-t_ufl) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Coefficient
    kappa = 1.0 + 0.8 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term
    f_ufl = -u_exact_ufl - ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    u_n = fem.Function(V)
    
    def initial_condition(x, t):
        return np.exp(-t) * np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    u_n.interpolate(lambda x_pts: initial_condition(x_pts, t0))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Backward Euler
    a = u * v * ufl.dx + dt_ufl * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_ufl * f_ufl * v * ufl.dx
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    def boundary_marker(x_pts):
        return np.full(x_pts.shape[1], True, dtype=bool)
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_pts: initial_condition(x_pts, t0))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    t = t0
    n_steps = int(np.round((t_end - t0) / dt))
    total_iterations = 0
    
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        
        u_bc.interpolate(lambda x_pts: initial_condition(x_pts, t))
        
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
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
        
    u_final = np.full((ny_out, nx_out), np.nan)
    u_initial = np.full((ny_out, nx_out), np.nan)
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        # Evaluate initial condition
        u_init_func = fem.Function(V)
        u_init_func.interpolate(lambda x_pts: initial_condition(x_pts, t0))
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial.flat[eval_map] = vals_init.flatten()
        
        # Evaluate final solution
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_final.flat[eval_map] = vals.flatten()
        
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_final,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    print("Solver finished.")
    print("iterations:", res["solver_info"]["iterations"])
