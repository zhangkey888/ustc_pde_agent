import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation with manufactured solution."""
    
    # Extract parameters from case_spec
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    
    # Time parameters with hardcoded defaults
    t_end = time_spec.get("t_end", 0.3)
    dt = time_spec.get("dt", 0.005)
    time_scheme = time_spec.get("scheme", "crank_nicolson")
    is_transient = True  # Force transient
    
    # Diffusion coefficient - check multiple locations
    epsilon = 1.0
    if "epsilon" in pde_spec:
        epsilon = float(pde_spec["epsilon"])
    if "coefficients" in pde_spec:
        epsilon = float(pde_spec["coefficients"].get("epsilon", epsilon))
    
    # Mesh and element parameters
    mesh_resolution = 80
    element_degree = 2
    
    # Output grid
    nx_out = case_spec.get("output", {}).get("nx", 70)
    ny_out = case_spec.get("output", {}).get("ny", 70)
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time constants
    t_param = fem.Constant(domain, ScalarType(0.0))       # for exact solution / BC
    t_new_c = fem.Constant(domain, ScalarType(0.0))       # t^{n+1} for source
    t_old_c = fem.Constant(domain, ScalarType(0.0))       # t^n for source
    
    # Manufactured solution: u = exp(-t) * sin(4*pi*x) * sin(3*pi*y)
    u_exact_ufl = ufl.exp(-t_param) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Source term: f = du/dt - eps*laplacian(u) + R(u)
    # With R(u) = u (linear reaction):
    # f = (-1 + 25*eps*pi^2 + 1) * exp(-t) * sin(4*pi*x) * sin(3*pi*y)
    # f = 25*eps*pi^2 * exp(-t) * sin(4*pi*x) * sin(3*pi*y)
    coeff_f = 25.0 * epsilon * pi**2
    
    f_new = coeff_f * ufl.exp(-t_new_c) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    f_old = coeff_f * ufl.exp(-t_old_c) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution functions
    u_n = fem.Function(V)    # solution at t^n
    u_new = fem.Function(V)  # solution at t^{n+1}
    
    # Set initial condition
    t_param.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Theta for time scheme
    if time_scheme == "backward_euler":
        theta = 1.0
    else:  # crank_nicolson
        theta = 0.5
    
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    theta_c = fem.Constant(domain, ScalarType(theta))
    one_m_theta_c = fem.Constant(domain, ScalarType(1.0 - theta))
    
    # Bilinear form (LHS):
    # (1/dt)*u*v + theta*(eps*grad(u)·grad(v) + u*v)
    a_form = (1.0 / dt_c * ufl.inner(u, v) * ufl.dx
              + theta_c * (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                           + ufl.inner(u, v) * ufl.dx))
    
    # Linear form (RHS):
    # (1/dt)*u_n*v - (1-theta)*(eps*grad(u_n)·grad(v) + u_n*v) + theta*f_new*v + (1-theta)*f_old*v
    L_form = (1.0 / dt_c * ufl.inner(u_n, v) * ufl.dx
              - one_m_theta_c * (eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
                                  + ufl.inner(u_n, v) * ufl.dx)
              + theta_c * ufl.inner(f_new, v) * ufl.dx
              + one_m_theta_c * ufl.inner(f_old, v) * ufl.dx)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    current_time = 0.0
    
    for step in range(n_steps):
        current_time += dt
        
        # Update time constants
        t_new_c.value = current_time
        t_old_c.value = current_time - dt
        
        # Update boundary conditions
        t_param.value = current_time
        u_bc.interpolate(u_exact_expr)
        
        # Reassemble matrix (BC rows change)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        b = petsc.assemble_vector(L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_new.x.petsc_vec)
        u_new.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_new.x.array[:]
        
        b.destroy()
    
    # === Evaluate solution on output grid ===
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate final solution
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_new.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Evaluate initial condition
    t_param.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up
    ksp.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson" if theta == 0.5 else "backward_euler",
        }
    }


if __name__ == "__main__":
    start = time_module.time()
    
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "time": {
                "t_end": 0.3,
                "dt": 0.005,
                "scheme": "crank_nicolson"
            },
            "epsilon": 1.0,
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 70, "ny": 70},
    }
    
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    x_out = np.linspace(0, 1, 70)
    y_out = np.linspace(0, 1, 70)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-0.3) * np.sin(4 * np.pi * X) * np.sin(3 * np.pi * Y)
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"L2 grid error: {error:.6e}")
    print(f"Max grid error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
