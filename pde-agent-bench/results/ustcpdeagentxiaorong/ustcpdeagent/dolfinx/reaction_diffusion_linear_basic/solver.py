import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def sample_on_grid(domain, u_func, nx, ny, bbox):
    """Sample a scalar FEM function on a uniform grid."""
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    return u_values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t_start_wall = time.time()

    # Extract case parameters
    pde = case_spec.get("pde", {})

    # Time parameters
    time_params = pde.get("time", {})
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.5))
    dt_suggested = float(time_params.get("dt", 0.01))
    time_scheme = time_params.get("scheme", "backward_euler")

    # Diffusion coefficient
    epsilon = float(pde.get("epsilon", 1.0))

    # Reaction coefficient
    reaction = pde.get("reaction", {})
    if isinstance(reaction, dict):
        sigma_val = float(reaction.get("sigma", 1.0))
    else:
        sigma_val = 1.0

    # Output grid
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {})
    nx_out = int(grid_spec.get("nx", 50))
    ny_out = int(grid_spec.get("ny", 50))
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    xmin, xmax, ymin, ymax = bbox

    # Solver parameters - optimized for accuracy within time budget
    mesh_res = 64
    element_degree = 2
    dt = dt_suggested / 2.0  # half suggested dt for better temporal accuracy
    ksp_type_str = "preonly"
    pc_type_str = "lu"
    rtol = 1e-10

    n_steps = int(round((t_end - t0) / dt))
    if n_steps < 1:
        n_steps = 1
    dt = (t_end - t0) / n_steps

    # Create mesh
    comm = MPI.COMM_WORLD
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                   cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates and time constant
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    pi = ufl.pi

    # Manufactured solution: u_exact = exp(-t) * sin(pi*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term: f = exp(-t)*sin(pi*x)*sin(pi*y) * (-1 + 2*eps*pi^2 + sigma)
    f_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * (
        -1.0 + 2.0 * epsilon * pi**2 + sigma_val
    )

    # Trial and test functions
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")
    t_const.value = t0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    # Store initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Backward Euler weak form
    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    sigma_c = fem.Constant(domain, ScalarType(sigma_val))

    a = (u_trial * v / dt_const + eps_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v))
         + sigma_c * u_trial * v) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx

    # Boundary conditions (u=0 on entire boundary for this manufactured solution)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant for this linear problem)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Create KSP solver with LU
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type_str)
    ksp.getPC().setType(pc_type_str)

    # Solution function
    u_sol = fem.Function(V, name="u")
    u_sol.x.array[:] = u_n.x.array[:]

    # Create RHS vector
    b = petsc.create_vector(V)

    # Time stepping
    total_iterations = 0
    t_current = t0

    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Sample solution on output grid
    u_grid = sample_on_grid(domain, u_sol, nx_out, ny_out, bbox)

    # Sample initial condition on output grid
    u_initial_grid = sample_on_grid(domain, u_initial_func, nx_out, ny_out, bbox)

    # Compute L2 error for verification
    t_const.value = t_end
    error_form = fem.form((u_sol - u_exact_ufl)**2 * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error at t={t_end}: {error_global:.6e}")

    wall_time = time.time() - t_start_wall
    print(f"Wall time: {wall_time:.2f}s")

    # Cleanup
    ksp.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "epsilon": 1.0,
            "reaction": {"type": "linear", "sigma": 1.0},
            "time": {
                "t0": 0.0,
                "t_end": 0.5,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }

    result = solve(case_spec)
    u_grid = result["u"]
    print(f"Output shape: {u_grid.shape}")
    print(f"u range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")

    # Check against exact solution at t=0.5
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.linspace(0.0, 1.0, 50)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.5) * np.sin(np.pi * XX) * np.sin(np.pi * YY)

    mask = ~np.isnan(u_grid)
    max_err = np.max(np.abs(u_grid[mask] - u_exact[mask]))
    l2_err = np.sqrt(np.mean((u_grid[mask] - u_exact[mask])**2))
    print(f"Max grid error: {max_err:.6e}")
    print(f"L2 grid error: {l2_err:.6e}")
    print(f"Solver info: {result['solver_info']}")
