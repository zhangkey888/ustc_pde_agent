import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]; ny_out = grid["ny"]
    bbox = grid["bbox"]

    t0 = 0.0
    t_end = 0.08
    dt = 0.001
    n_steps = int(round((t_end - t0) / dt))
    
    kappa = 1.0
    
    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Manufactured: u = exp(-t)*exp(5x)*sin(pi*y)
    # du/dt = -u
    # -kappa*lap(u) = -kappa*(25*u - pi^2*u) = -kappa*(25-pi^2)*u
    # f = du/dt - kappa*lap(u) = -u - kappa*(25-pi^2)*u = (-1 - kappa*(25 - pi^2))*u
    # Wait: lap(u) = 25*u - pi^2*u, so -kappa*lap(u) = -kappa*(25-pi^2)*u
    # f = -u + (-kappa*(25-pi^2)*u)? No: equation is du/dt - kappa*lap(u) = f
    # du/dt = -u; -kappa*lap(u) = -kappa*(25-pi^2)*u
    # f = -u - kappa*(25 - pi^2)*u = -(1 + kappa*(25 - pi^2))*u
    
    u_exact_expr = ufl.exp(-t_const) * ufl.exp(5.0*x[0]) * ufl.sin(ufl.pi*x[1])
    f_expr = -(1.0 + kappa*(25.0 - ufl.pi**2)) * u_exact_expr
    
    # BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    def bc_func(x, t):
        return np.exp(-t) * np.exp(5.0*x[0]) * np.sin(np.pi*x[1])
    
    u_bc.interpolate(lambda x: bc_func(x, t0))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: bc_func(x, t0))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler: (u - u_n)/dt - kappa*lap(u) = f
    # Weak: (u, v)/dt + kappa*(grad u, grad v) = (u_n, v)/dt + (f, v)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u*v/dt_c + kappa_c*ufl.inner(ufl.grad(u), ufl.grad(v)))*ufl.dx
    L = (u_n*v/dt_c + f_expr*v)*ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=200)
    
    u_sol = fem.Function(V)
    
    # Save initial for u_initial output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: bc_func(x, t0))
    
    total_iters = 0
    t = t0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        u_bc.interpolate(lambda x: bc_func(x, t))
        
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
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cc = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cc, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k, 0] if vals.ndim > 1 else vals[k]
    
    u_grid = u_vals.reshape(ny_out, nx_out)
    
    # Initial condition sampled
    u_init_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals0 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_init_vals[idx] = vals0[k, 0] if vals0.ndim > 1 else vals0[k]
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
        "pde": {"time": {"t_end": 0.08, "dt": 0.008}}
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    # Compare with exact
    grid = case_spec["output"]["grid"]
    xs = np.linspace(0, 1, grid["nx"])
    ys = np.linspace(0, 1, grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.08
    u_ex = np.exp(-t_end) * np.exp(5*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((result["u"] - u_ex)**2))
    print(f"Time: {elapsed:.3f}s, RMS error: {err:.6e}")
    print(f"Max error: {np.max(np.abs(result['u'] - u_ex)):.6e}")
    print(f"Info: {result['solver_info']}")
