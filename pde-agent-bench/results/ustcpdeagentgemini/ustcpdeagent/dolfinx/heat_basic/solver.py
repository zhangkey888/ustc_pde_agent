import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    output_grid = case_spec.get("output", {}).get("grid", {})
    nx = output_grid.get("nx", 50)
    ny = output_grid.get("ny", 50)
    bbox = output_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    t0 = 0.0
    t_end = 0.1
    
    # We want error <= 1.32e-03 and time <= 10.5s.
    # 64x64 degree 2 with dt=0.005 should be very fast and accurate.
    mesh_res = 64
    degree = 2
    dt = 0.005
    n_steps = int(np.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    def exact_expr_ufl(t_val):
        return ufl.exp(-t_val) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_n.interpolate(fem.Expression(exact_expr_ufl(PETSc.ScalarType(t0)), V.element.interpolation_points))
    
    # Save initial state
    u_init_grid = None
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(exact_expr_ufl(PETSc.ScalarType(t0)), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # f = du/dt - kappa * Delta u
    # u = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -u, Delta u = -2*pi^2 * u
    # f = (-1 + 2*pi^2) * u
    f_ufl = (-1.0 + 2.0 * ufl.pi**2) * exact_expr_ufl(t_ufl)
    
    # Backward Euler weak form:
    # (u - u_n)/dt - Delta u = f
    # -> (u, v) + dt*(grad(u), grad(v)) = (u_n, v) + dt*(f, v)
    a = ufl.inner(u, v) * ufl.dx + dt_ufl * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_ufl * ufl.inner(f_ufl, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp_type = "preonly"
    pc_type = "lu"
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t = t0
    iterations = 0
    
    for i in range(n_steps):
        t += dt
        t_ufl.value = t
        
        # update BC
        u_bc.interpolate(fem.Expression(exact_expr_ufl(PETSc.ScalarType(t)), V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        u_n.x.array[:] = u_sol.x.array[:]
        iterations += solver.getIterationNumber()
        
    def sample_func(func, nx, ny, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        X, Y = np.meshgrid(xs, ys)
        pts = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
        
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
                
        u_values = np.full(pts.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        return u_values.reshape((ny, nx))

    u_grid = sample_func(u_sol, nx, ny, bbox)
    u_init_grid = sample_func(u_initial_func, nx, ny, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": iterations,
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
    spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    t_start = time.time()
    res = solve(spec)
    t_end = time.time()
    
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.linspace(0.0, 1.0, 50)
    X, Y = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.nanmax(np.abs(res["u"] - u_exact))
    print(f"Max error: {error}")
    print(f"Wall time: {t_end - t_start} s")

