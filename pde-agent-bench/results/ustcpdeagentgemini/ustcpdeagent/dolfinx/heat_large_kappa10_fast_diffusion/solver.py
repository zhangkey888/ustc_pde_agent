import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    t0 = 0.0
    t_end = 0.05
    dt = 0.0025
    kappa = 10.0
    
    # Grid parameters
    mesh_resolution = 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
                                   [mesh_resolution, mesh_resolution], 
                                   cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary Conditions (u=0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    
    u_init_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_init_expr_form = fem.Expression(u_init_expr, V.element.interpolation_points)
    u_n.interpolate(u_init_expr_form)
    
    u_initial_copy = fem.Function(V)
    u_initial_copy.x.array[:] = u_n.x.array[:]
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    t_const = fem.Constant(domain, PETSc.ScalarType(t0 + dt))
    
    # Exact solution for f
    u_exact = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = (-1.0 + 10.0 * 2.0 * ufl.pi**2) * u_exact
    
    # Backward Euler
    F = u * v * ufl.dx - u_n * v * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt * f * v * ufl.dx
    a, L = ufl.lhs(F), ufl.rhs(F)
    
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
    
    n_steps = int(np.round((t_end - t0) / dt))
    total_iterations = 0
    
    for i in range(n_steps):
        t_const.value = t0 + (i + 1) * dt
        
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluation
    def sample_on_grid(u_func, nx, ny, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
        
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
        
        points_on_proc = []
        cells = []
        eval_map = []
        for i, pt in enumerate(pts):
            links = colliding.links(i)
            if len(links) > 0:
                points_on_proc.append(pt)
                cells.append(links[0])
                eval_map.append(i)
                
        u_vals = np.full((nx * ny,), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
            u_vals[eval_map] = vals.flatten()
            
        return u_vals.reshape(ny, nx)
    
    u_grid = sample_on_grid(u_sol, nx_out, ny_out, bbox)
    u_init_grid = sample_on_grid(u_initial_copy, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
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
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    t_start = time.time()
    result = solve(case_spec)
    print("Done. Iterations:", result["solver_info"]["iterations"])
    
    u_grid = result["u"]
    nx, ny = 64, 64
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)
    exact = np.exp(-0.05) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    error = np.linalg.norm(u_grid - exact) / np.linalg.norm(exact)
    print(f"Error: {error:.4e}")
    print(f"Time: {time.time() - t_start:.2f}s")
