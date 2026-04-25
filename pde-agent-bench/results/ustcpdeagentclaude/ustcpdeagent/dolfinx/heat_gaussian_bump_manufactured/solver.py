import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid output spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Params - use smaller dt for better accuracy
    kappa = 1.0
    t0 = 0.0
    t_end = 0.1
    dt = 0.002
    n_steps = int(round((t_end - t0) / dt))
    
    # P2 elements with balanced mesh resolution
    mesh_res = 140
    degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Exact solution u = exp(-t)*exp(-40*((x-0.5)^2 + (y-0.5)^2))
    def u_exact_ufl(t_c):
        return ufl.exp(-t_c) * ufl.exp(-40.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
    
    # Source term: f = du/dt - kappa*laplacian(u)
    # u = exp(-t)*G(x,y), G = exp(-40*r2)
    # du/dt = -u
    # grad G = G * (-80*(x-0.5), -80*(y-0.5))
    # lap G = G * (80^2 * ((x-0.5)^2+(y-0.5)^2) - 160)
    r2 = (x[0]-0.5)**2 + (x[1]-0.5)**2
    G = ufl.exp(-40.0 * r2)
    lap_G = G * (6400.0 * r2 - 160.0)
    u_ex = ufl.exp(-t_const) * G
    # du/dt = -exp(-t)*G
    dudt = -ufl.exp(-t_const) * G
    lap_u = ufl.exp(-t_const) * lap_G
    f_expr = dudt - kappa * lap_u
    
    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(ufl.exp(-40.0 * r2), V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Boundary condition function (time-dependent)
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form - backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_const * f_expr) * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12)
    
    u_sol = fem.Function(V)
    
    total_iters = 0
    t = t0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        u_bc.interpolate(u_bc_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)
    
    u_grid = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[idx_map] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Initial condition sample
    u_init_grid = np.exp(-40.0 * ((XX-0.5)**2 + (YY-0.5)**2))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
    }
    t0 = time.time()
    result = solve(case_spec)
    wall = time.time() - t0
    
    # Check error
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.exp(-40.0 * ((XX-0.5)**2 + (YY-0.5)**2))
    err = np.sqrt(np.mean((result["u"] - u_ex)**2))
    print(f"L2_ERROR: {err:.6e}")
    print(f"WALL_TIME: {wall:.3f}")
    print(f"iters: {result['solver_info']['iterations']}")
