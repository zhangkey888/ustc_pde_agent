import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    coeffs = pde_spec.get("coefficients", {})
    
    t_end = time_spec.get("t_end", 0.12)
    dt_val = time_spec.get("dt", 0.006)
    scheme = time_spec.get("scheme", "backward_euler")
    kappa_val = coeffs.get("kappa", 0.5)
    
    N = 32
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    u_exact_ufl = ufl.exp(-10.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    f_ufl = ufl.exp(-10.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-10.0 + 2.0 * kappa_val * ufl.pi**2)
    
    kappa_c = fem.Constant(domain, ScalarType(kappa_val))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    
    u_n.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    a = (u / dt_c) * v * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_ufl * v * ufl.dx
    
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func = fem.Function(V)
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Create vector using function space V
    b_vec = petsc.create_vector(V)
    
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    ksp.setUp()
    
    t = 0.0
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        u_bc_func.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        ksp.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    ksp.destroy()
    A.destroy()
    b_vec.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.12,
                "dt": 0.006,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": 0.5
            }
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    
    t_end_val = 0.12
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    u_exact = np.exp(-10 * t_end_val) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Target: error <= 3.53e-01")
