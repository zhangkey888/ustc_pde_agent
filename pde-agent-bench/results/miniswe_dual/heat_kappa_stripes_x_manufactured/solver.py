import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t_end = time_spec.get("t_end", 0.1)
    dt_val = time_spec.get("dt", 0.01)
    scheme = time_spec.get("scheme", "backward_euler")
    
    N = 48
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    kappa = 1.0 + 0.5 * ufl.sin(6.0 * ufl.pi * x[0])
    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    dudt = -u_exact_ufl
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa * grad_u_exact)
    f = dudt - div_kappa_grad_u
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    
    t_c.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    n_steps = int(np.round(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_const = fem.Constant(domain, PETSc.ScalarType(actual_dt))
    
    a = (u / dt_const) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (u_n / dt_const) * v * ufl.dx + f * v * ufl.dx
    
    a_compiled = fem.form(a)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(V)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    total_iterations = 0
    
    for step in range(n_steps):
        current_t = (step + 1) * actual_dt
        t_c.value = current_t
        
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    t_c.value = 0.0
    u_init_func = fem.Function(V)
    u_init_expr2 = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_init_func.interpolate(u_init_expr2)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

if __name__ == "__main__":
    case_spec = {"pde": {"time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}}}
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
