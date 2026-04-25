import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Solver parameters
    spatial_res = 80
    degree = 2
    t0 = 0.0
    t_end = 0.1
    dt = 0.0025
    
    # Create mesh
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [spatial_res, spatial_res], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Initial condition
    def initial_cond(x_pt):
        return np.sin(np.pi*x_pt[0]) * np.sin(np.pi*x_pt[1])
        
    u_n.interpolate(initial_cond)
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    # Boundary conditions: zero everywhere (from u = exp(-t)*sin(pi*x)*sin(pi*y))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_pt: np.full(x_pt.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Source term
    t_c = fem.Constant(domain, PETSc.ScalarType(t0))
    # f = u_t - div(kappa grad(u))
    # u = exp(-t) sin(pi x) sin(pi y)
    # u_t = -exp(-t) sin(pi x) sin(pi y)
    # div(grad(u)) = -2 pi^2 exp(-t) sin(pi x) sin(pi y)
    # f = (-1 + 2 pi^2) exp(-t) sin(pi x) sin(pi y)
    f = (2.0*ufl.pi**2 - 1.0) * ufl.exp(-t_c) * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Theta scheme formulation (backward Euler -> theta = 1)
    # (u - u_n)/dt - div(grad(u)) = f
    # -> u - dt*div(grad(u)) = u_n + dt*f
    
    F = u*v*ufl.dx + dt*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - u_n*v*ufl.dx - dt*f*v*ufl.dx
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
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    num_steps = int(np.round((t_end - t0) / dt))
    iterations = 0
    t = t0
    
    for i in range(num_steps):
        t += dt
        t_c.value = t
        
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
        if solver.getIterationNumber() == 0:
            iterations += 1

    # Evaluate exact solution at t_end
    # exact = exp(-t_end)*sin(pi*x)*sin(pi*y)
    
    # Interpolate to output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_out = np.full(len(points), np.nan)
    u_init_out = np.full(len(points), np.nan)
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cls_array = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_sol.eval(pts_array, cls_array)
        u_out[eval_map] = u_vals.flatten()
        u_init_vals = u_initial.eval(pts_array, cls_array)
        u_init_out[eval_map] = u_init_vals.flatten()
        
    u_grid = u_out.reshape((ny_out, nx_out))
    u_init_grid = u_init_out.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": spatial_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iterations,
        "dt": dt,
        "n_steps": num_steps,
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
    import time
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    
    u = res["u"]
    
    # Exact solution at t=0.1
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    exact = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    err = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    print(f"Time: {t1-t0:.3f} s")
    print(f"Error: {err:.3e}")
