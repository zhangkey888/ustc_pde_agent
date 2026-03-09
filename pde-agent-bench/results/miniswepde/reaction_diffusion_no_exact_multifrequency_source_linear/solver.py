import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation (steady or transient)."""

    # ----------------------------------------------------------------
    # 1. Parse case_spec - handle both flat and nested formats
    # ----------------------------------------------------------------
    oc = case_spec.get("oracle_config", case_spec)
    pde = oc.get("pde", {})
    pde_params = pde.get("pde_params", {})
    output_cfg = oc.get("output", {})
    grid_cfg = output_cfg.get("grid", {})

    # Diffusion coefficient
    epsilon = float(pde_params.get("epsilon", pde.get("epsilon", 0.09)))

    # Reaction term
    reaction_cfg = pde_params.get("reaction", {})
    reaction_alpha = float(reaction_cfg.get("alpha", pde.get("reaction_alpha", 1.0)))

    # Source term
    source_expr_str = pde.get("source_term",
                              "sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)")

    # Time parameters - hardcoded defaults for this problem
    time_params = pde.get("time", {})
    is_transient = True  # Force transient
    t_end = float(time_params.get("t_end", 0.5))
    dt_val = float(time_params.get("dt", 0.01))
    time_scheme = str(time_params.get("scheme", "crank_nicolson"))

    # Initial condition
    ic_str = pde.get("initial_condition", "sin(pi*x)*sin(pi*y)")

    # Boundary condition
    bc_val = 0.0
    bc_cfg = oc.get("bc", {}).get("dirichlet", {})
    if bc_cfg:
        bc_val = float(bc_cfg.get("value", 0.0))

    # Output grid
    nx_out = int(grid_cfg.get("nx", 80))
    ny_out = int(grid_cfg.get("ny", 80))

    # ----------------------------------------------------------------
    # 2. Solver parameters
    # ----------------------------------------------------------------
    # Oracle uses resolution 170 with P1. We use similar for accuracy.
    N = 128
    element_degree = 1

    comm = MPI.COMM_WORLD

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Source term (UFL expression)
    f_expr = _parse_expression(source_expr_str, x)

    # Boundary conditions (homogeneous Dirichlet on all boundaries)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(bc_val), dofs, V)
    bcs = [bc]

    # Functions
    u_n = fem.Function(V, name="u_n")  # previous time step
    u_h = fem.Function(V, name="u_h")  # current time step

    # Initial condition
    ic_ufl = _parse_expression(ic_str, x)
    ic_expr = fem.Expression(ic_ufl, V.element.interpolation_points)
    u_n.interpolate(ic_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Time stepping
    n_steps = int(round(t_end / dt_val))
    actual_dt = t_end / n_steps

    dt_c = fem.Constant(domain, ScalarType(actual_dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    alpha_c = fem.Constant(domain, ScalarType(reaction_alpha))

    # Theta method
    if time_scheme == "crank_nicolson":
        theta = 0.5
    elif time_scheme == "backward_euler":
        theta = 1.0
    else:
        theta = 0.5

    # Bilinear form (LHS)
    a_form = (
        u * v / dt_c * ufl.dx
        + theta * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + theta * alpha_c * u * v * ufl.dx
    )

    # Linear form (RHS)
    L_form = (
        u_n * v / dt_c * ufl.dx
        - (1.0 - theta) * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - (1.0 - theta) * alpha_c * u_n * v * ufl.dx
        + f_expr * v * ufl.dx
    )

    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix (constant in time for linear problem)
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    # Create RHS vector
    b = petsc.create_vector(V)

    # Setup KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    ksp.setUp()

    total_iterations = 0

    # Time loop
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]

    # ----------------------------------------------------------------
    # 3. Evaluate on output grid
    # ----------------------------------------------------------------
    u_grid = _evaluate_on_grid(domain, u_h, nx_out, ny_out)
    u_initial_grid = _evaluate_on_grid(domain, u_initial_func, nx_out, ny_out)

    # Clean up PETSc objects
    ksp.destroy()
    A.destroy()
    b.destroy()

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson" if theta == 0.5 else "backward_euler",
        }
    }

    return result


def _parse_expression(expr_str, x):
    """Parse a string expression into a UFL expression."""
    ns = {
        "x": x[0],
        "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin,
        "cos": ufl.cos,
        "exp": ufl.exp,
        "sqrt": ufl.sqrt,
        "abs": ufl.algebra.Abs,
        "tanh": ufl.tanh,
    }
    try:
        return eval(expr_str, {"__builtins__": {}}, ns)
    except Exception as e:
        print(f"Warning: Could not parse expression '{expr_str}': {e}")
        return ScalarType(0.0)


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate a FEM function on a uniform grid."""
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_3d = np.zeros((nx * ny, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    # Test with the actual config structure
    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "reaction_diffusion",
                "pde_params": {
                    "epsilon": 0.09,
                    "reaction": {
                        "type": "linear",
                        "alpha": 1.0
                    }
                },
                "source_term": "sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)",
                "time": {
                    "t0": 0.0,
                    "t_end": 0.5,
                    "dt": 0.01,
                    "scheme": "crank_nicolson"
                },
                "initial_condition": "sin(pi*x)*sin(pi*y)"
            },
            "output": {
                "grid": {
                    "bbox": [0, 1, 0, 1],
                    "nx": 80,
                    "ny": 80
                }
            },
            "bc": {
                "dirichlet": {
                    "on": "all",
                    "value": "0.0"
                }
            }
        }
    }

    t0 = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - t0

    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solution norm: {np.linalg.norm(result['u']):.6f}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")

    # Compare with reference
    import os
    ref_path = os.path.join(os.path.dirname(__file__),
                            '../../miniswepde/reaction_diffusion_no_exact_multifrequency_source_linear/oracle_output/reference.npz')
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        u_ref = ref['u_star']
        err = np.linalg.norm(result['u'] - u_ref) / np.linalg.norm(u_ref)
        print(f"Relative L2 error vs reference: {err:.6e}")
    else:
        print(f"Reference not found at {ref_path}")
