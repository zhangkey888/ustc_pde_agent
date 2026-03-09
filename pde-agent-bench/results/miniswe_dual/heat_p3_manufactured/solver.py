import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """Solve the transient heat equation with manufactured solution."""
    
    # ---- Extract parameters from case_spec or use defaults ----
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    coeffs = pde_spec.get("coefficients", {})
    
    # Time parameters - hardcoded defaults from problem description
    t_end = float(time_spec.get("t_end", 0.08))
    dt_val = float(time_spec.get("dt", 0.008))
    scheme = time_spec.get("scheme", "backward_euler")
    
    # Diffusivity
    kappa = float(coeffs.get("kappa", 1.0))
    
    # Spatial parameters
    N = 64
    degree = 2
    
    # ---- Create mesh ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # ---- Spatial coordinate and time ----
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a Constant that we update
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # ---- Manufactured solution ----
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(pi*x)*sin(2*pi*y)
    # laplacian(u) = -5*pi^2*exp(-t)*sin(pi*x)*sin(2*pi*y)
    # -kappa*laplacian(u) = 5*kappa*pi^2*exp(-t)*sin(pi*x)*sin(2*pi*y)
    # f = (-1 + 5*kappa*pi^2)*exp(-t)*sin(pi*x)*sin(2*pi*y)
    f_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1]) * (-1.0 + 5.0 * kappa * pi**2)
    
    # ---- Trial and test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # ---- Previous solution ----
    u_n = fem.Function(V, name="u_n")
    
    # ---- Initial condition ----
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2 * np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2 * np.pi * X[1]))
    
    # ---- Backward Euler time discretization ----
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_ufl * v) * ufl.dx
    
    # ---- Boundary conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # ---- Solution function ----
    u_sol = fem.Function(V, name="u_sol")
    
    # ---- Assemble matrix ----
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # ---- Create RHS vector ----
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # ---- Setup KSP solver ----
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # ---- Expression for BC update ----
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # ---- Time stepping ----
    t = 0.0
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        # Update boundary condition
        bc_func.interpolate(bc_expr)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        
        # Apply lifting and BCs
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # ---- Evaluate on 50x50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # ---- Cleanup ----
    solver.destroy()
    A.destroy()
    
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
            "time_scheme": scheme,
        }
    }


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u = result['u']
    info = result['solver_info']
    
    print(f'Shape: {u.shape}')
    print(f'NaN count: {np.isnan(u).sum()}')
    print(f'Min: {np.nanmin(u):.6e}, Max: {np.nanmax(u):.6e}')
    print(f'Wall time: {elapsed:.3f}s')
    
    # Compute exact solution at t_end
    t_end = 0.08
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    
    error = np.sqrt(np.mean((u - u_exact)**2))
    max_err = np.max(np.abs(u - u_exact))
    print(f'RMS error: {error:.6e}')
    print(f'Max error: {max_err:.6e}')
    print(f'Solver info: {info}')
