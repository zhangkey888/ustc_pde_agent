import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid output spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025
    kappa_val = 1.0
    
    # Mesh - increased for higher accuracy
    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    # Exact solution: u = exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y))
    def u_exact_ufl(t_c):
        return ufl.exp(-t_c) * ufl.sin(3*ufl.pi*(x[0]+x[1])) * ufl.sin(ufl.pi*(x[0]-x[1]))
    
    # Compute f = du/dt - kappa * laplacian(u)
    # du/dt = -u
    u_ex = u_exact_ufl(t_const)
    dudt = -u_ex  # derivative w.r.t. t
    laplace_u = ufl.div(ufl.grad(u_ex))
    f_expr = dudt - kappa * laplace_u
    
    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(u_exact_ufl(fem.Constant(domain, PETSc.ScalarType(t0))),
                                  V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Save initial for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # BC function
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    a = u*v*ufl.dx + dt_c*kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = (u_n + dt_c*f_expr)*v*ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-12, atol=1e-14)
    
    u_sol = fem.Function(V)
    
    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0
    t = t0
    
    for step in range(n_steps):
        t_new = t + dt_val
        t_const.value = t_new
        u_bc.interpolate(bc_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    
    bb = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(bb, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)
    
    u_vals = np.zeros(nx_out*ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[idx_map] = vals.flatten()
    
    u_grid = u_vals.reshape(ny_out, nx_out)
    
    # Also sample initial
    u_init_vals = np.zeros(nx_out*ny_out)
    if len(points_on_proc) > 0:
        vals_i = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[idx_map] = vals_i.flatten()
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    
    # Compute error against exact solution
    u_grid = res["u"]
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.1
    u_ex = np.exp(-t_end) * np.sin(3*np.pi*(XX+YY)) * np.sin(np.pi*(XX-YY))
    err = np.sqrt(np.mean((u_grid - u_ex)**2))
    maxerr = np.max(np.abs(u_grid - u_ex))
    print(f"L2 err: {err:.6e}, Max err: {maxerr:.6e}")
    print(f"Solver info: {res['solver_info']}")
