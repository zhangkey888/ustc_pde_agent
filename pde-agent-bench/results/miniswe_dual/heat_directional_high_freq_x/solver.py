import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with backward Euler."""
    
    # Parse parameters
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    coeffs = pde.get("coefficients", {})
    
    kappa = float(coeffs.get("kappa", 1.0))
    t_end = float(time_params.get("t_end", 0.08))
    dt = float(time_params.get("dt", 0.004))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Solver parameters - high freq sin(8*pi*x) needs fine mesh
    # 8 full waves in [0,1], need ~16 pts per wave minimum -> 128
    # With degree=2, N=96 should be sufficient
    N = 96
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant (updated each step)
    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    
    # Manufactured solution: u = exp(-t)*sin(8*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term derivation:
    # u = exp(-t)*sin(8*pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(8*pi*x)*sin(pi*y) = -u
    # d2u/dx2 = -64*pi^2 * u
    # d2u/dy2 = -pi^2 * u
    # laplacian(u) = -(64*pi^2 + pi^2)*u = -65*pi^2 * u
    # f = du/dt - kappa*laplacian(u) = -u + kappa*65*pi^2*u = u*(-1 + 65*kappa*pi^2)
    f_ufl = ufl.exp(-t_const) * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (
        -1.0 + 65.0 * kappa * ufl.pi**2
    )
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution functions
    u_n = fem.Function(V, name="u_n")  # previous time step
    u_h = fem.Function(V, name="u_h")  # current time step
    
    # Initial condition: u(x,0) = sin(8*pi*x)*sin(pi*y)
    t_const.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    # Store initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler weak form:
    # (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # => (u/dt)*v*dx + kappa*grad(u)·grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    a = (u / dt_const) * v * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # Time stepping
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC from exact solution at current time
        u_bc.interpolate(u_exact_expr)
        
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Re-assemble matrix with updated BCs
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
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
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
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
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.08, "dt": 0.004, "scheme": "backward_euler"},
        },
        "domain": {},
        "output": {"nx": 50, "ny": 50},
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    u_grid = result["u"]
    
    # Exact solution at t_end
    t_end = 0.08
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(8 * np.pi * X) * np.sin(np.pi * Y)
    
    rms_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.2f}s")
    print(f"RMS error: {rms_error:.6e}")
    print(f"Linf error: {linf_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
