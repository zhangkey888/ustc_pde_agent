import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation using backward Euler."""
    
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = coeffs.get("kappa", 1.0)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_val = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    N = 64
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    kappa_c = fem.Constant(domain, ScalarType(kappa_val))
    
    f_ufl = (-1.0 + 2.0 * kappa_val * ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    a = u * v * ufl.dx + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_ufl * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector from function space
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t = 0.0
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value.fill(t)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    
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
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts, cls)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
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
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
        },
        "domain": {},
        "output": {"nx": 50, "ny": 50},
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    u_grid = result["u"]
    t_end_val = 0.1
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end_val) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 grid error: {error:.6e}")
    print(f"Max grid error: {max_error:.6e}")
    print(f"Target error: <= 2.98e-03")
    print(f"Target time: <= 15.022s")
    print(f"PASS accuracy: {error <= 2.98e-3}")
    print(f"PASS time: {elapsed <= 15.022}")
