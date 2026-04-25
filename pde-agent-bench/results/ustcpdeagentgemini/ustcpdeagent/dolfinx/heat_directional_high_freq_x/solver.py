import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    t0 = 0.0
    t_end = 0.08
    dt = 0.002
    
    mesh_res = 80
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    
    u_exact = ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = (65 * ufl.pi**2 - 1) * u_exact
    
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_pts: np.full(x_pts.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = ufl.inner(u - u_n, v) * ufl.dx + dt_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_const * ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    n_steps = 0
    current_t = t0
    
    while current_t < t_end - 1e-10:
        current_t += dt
        n_steps += 1
        t.value = current_t
        
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += 1
        u_n.x.array[:] = u_sol.x.array
    
    def sample_func(func, nx, ny, bbox):
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
                
        u_values = np.zeros(pts.shape[1])
        if len(points_on_proc) > 0:
            vals = func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        return u_values.reshape((ny, nx))

    u_grid = sample_func(u_sol, nx_out, ny_out, bbox)
    u_initial_grid = sample_func(u_initial_func, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    start = time.time()
    res = solve(case_spec)
    end = time.time()
    
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.08
    u_exact = np.exp(-t_end) * np.sin(8 * np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.linalg.norm(res["u"] - u_exact) / np.linalg.norm(u_exact)
    print(f"Time: {end - start:.2f} s")
    print(f"Error: {error:.4e}")
