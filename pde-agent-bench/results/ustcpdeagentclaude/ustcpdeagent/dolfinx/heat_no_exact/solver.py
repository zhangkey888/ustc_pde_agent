import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025
    kappa = 1.0
    
    # Mesh and space
    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # BCs: u=0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial condition
    x = ufl.SpatialCoordinate(domain)
    u_n = fem.Function(V)
    u0_expr = fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u0_expr)
    
    # Save initial
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Source term
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Variational form: backward Euler
    # (u - u_n)/dt - kappa * div(grad(u)) = f
    # => u/dt*v + kappa*grad(u).grad(v) = u_n/dt*v + f*v
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u / dt_c) * v * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    rtol = 1e-10
    solver.setTolerances(rtol=rtol)
    
    u_h = fem.Function(V)
    
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0
    
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    # Sample on grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(ny, nx)
    
    # Initial
    u_init_grid = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals0 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_grid[eval_map] = vals0.flatten()
    u_init_grid = u_init_grid.reshape(ny, nx)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": True}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{result['u'].min():.4f}, {result['u'].max():.4f}]")
    print(f"Info: {result['solver_info']}")
