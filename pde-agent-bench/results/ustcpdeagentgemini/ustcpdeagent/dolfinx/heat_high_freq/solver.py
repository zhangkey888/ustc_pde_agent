import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    mesh_res = 128
    degree = 2
    dt = 0.005
    t_end = 0.1
    t = 0.0
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    u_initial = u_n.x.array.copy()
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def exact_solution(x, current_t):
        return np.exp(-current_t) * np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
        
    u_bc.interpolate(lambda x: exact_solution(x, t+dt))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Forms
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t+dt))
    f = (32 * ufl.pi**2 - 1) * ufl.exp(-t_ufl) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = u * v * ufl.dx + dt_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    n_steps = int(np.round(t_end / dt))
    
    total_iterations = 0
    
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        u_bc.interpolate(lambda x: exact_solution(x, t))
        
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
        
        u_n.x.array[:] = u_sol.x.array
        total_iterations += solver.getIterationNumber()
        
    # Interpolate to grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape(ny, nx)
    
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_initial
    
    u_initial_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals_initial = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_values[eval_map] = vals_initial.flatten()
    u_initial_grid = u_initial_values.reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    import time
    t0 = time.time()
    res = solve(case_spec)
    print("Time taken:", time.time() - t0)
    print("Iterations:", res["solver_info"]["iterations"])
