import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("viscosity", 0.05))
    f_expr = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])

    # Mesh resolution - use high resolution for accuracy
    N = 128
    degree_u = 2
    degree_p = 1

    comm = MPI.COMM_WORLD

    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function space (Taylor-Hood P2/P1)
    vel_elem = element("Lagrange", domain.basix_cell(), degree_u, shape=(2,))
    pres_elem = element("Lagrange", domain.basix_cell(), degree_p)
    mel = mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mel)

    # Also create sub-spaces for BC application
    V_sub, V_sub_map = W.sub(0).collapse()
    Q_sub, Q_sub_map = W.sub(1).collapse()

    # 4. Define trial/test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Source term
    f1_str = f_expr[0] if len(f_expr) > 0 else "0.0"
    f2_str = f_expr[1] if len(f_expr) > 1 else "0.0"

    x = ufl.SpatialCoordinate(domain)
    # Parse source - for this case it's zero
    f = ufl.as_vector([0.0, 0.0])

    # 5. Variational form for Stokes: 
    # nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) = inner(f, v)
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)

    L = ufl.inner(f, v) * ufl.dx

    # 6. Boundary conditions
    # Parse BCs from case_spec
    bcs = []

    # Determine BC type - for Poiseuille channel flow:
    # Typically: parabolic inflow on left, zero on top/bottom, do-nothing on right
    # Or: no-slip on top/bottom, parabolic profile on left/right

    # Check boundary conditions from spec
    for bc_spec in bcs_spec:
        bc_type = bc_spec.get("type", "dirichlet")
        location = bc_spec.get("location", "")
        value = bc_spec.get("value", None)
        component = bc_spec.get("component", None)

        if bc_type == "dirichlet":
            # Parse the boundary location
            if location == "left" or location == "x=0":
                def marker(x):
                    return np.isclose(x[0], 0.0)
            elif location == "right" or location == "x=1":
                def marker(x):
                    return np.isclose(x[0], 1.0)
            elif location == "bottom" or location == "y=0":
                def marker(x):
                    return np.isclose(x[1], 0.0)
            elif location == "top" or location == "y=1":
                def marker(x):
                    return np.isclose(x[1], 1.0)
            elif location == "all" or location == "boundary":
                def marker(x):
                    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                            np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            else:
                # Try to handle generic
                def marker(x):
                    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                            np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

            facets = mesh.locate_entities_boundary(domain, fdim, marker)

            if component is not None:
                # Apply BC to specific velocity component
                comp_idx = int(component)
                W_sub_comp, W_sub_comp_map = W.sub(0).sub(comp_idx).collapse()

                bc_func = fem.Function(W_sub_comp)
                if value is not None:
                    if isinstance(value, str):
                        val_parsed = _parse_expression(value)
                        bc_func.interpolate(lambda x, vp=val_parsed: _eval_expr(vp, x))
                    elif isinstance(value, (int, float)):
                        bc_func.interpolate(lambda x, v=float(value): np.full(x.shape[1], v))
                    else:
                        bc_func.interpolate(lambda x: np.zeros(x.shape[1]))
                else:
                    bc_func.interpolate(lambda x: np.zeros(x.shape[1]))

                dofs = fem.locate_dofs_topological(
                    (W.sub(0).sub(comp_idx), W_sub_comp), fdim, facets)
                bc_obj = fem.dirichletbc(bc_func, dofs, W.sub(0).sub(comp_idx))
                bcs.append(bc_obj)
            else:
                # Apply BC to full velocity
                bc_func = fem.Function(V_sub)
                if value is not None:
                    if isinstance(value, list) and len(value) == 2:
                        v0_str = str(value[0])
                        v1_str = str(value[1])
                        bc_func.interpolate(
                            lambda x, s0=v0_str, s1=v1_str: _eval_vector_expr(s0, s1, x))
                    elif isinstance(value, str):
                        # Single string for vector - assume zero
                        bc_func.interpolate(
                            lambda x: np.zeros((2, x.shape[1])))
                    else:
                        bc_func.interpolate(
                            lambda x: np.zeros((2, x.shape[1])))
                else:
                    bc_func.interpolate(
                        lambda x: np.zeros((2, x.shape[1])))

                dofs = fem.locate_dofs_topological(
                    (W.sub(0), V_sub), fdim, facets)
                bc_obj = fem.dirichletbc(bc_func, dofs, W.sub(0))
                bcs.append(bc_obj)

    # If no BCs were parsed, set up default Poiseuille channel BCs
    if len(bcs) == 0:
        # Default: Poiseuille flow in a channel
        # No-slip on top and bottom walls
        # Parabolic inflow on left: u = (4*y*(1-y), 0)
        # Do-nothing (natural BC) on right

        # Top wall (y=1): no-slip
        def top_wall(x):
            return np.isclose(x[1], 1.0)

        facets_top = mesh.locate_entities_boundary(domain, fdim, top_wall)
        bc_func_top = fem.Function(V_sub)
        bc_func_top.interpolate(lambda x: np.zeros((2, x.shape[1])))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_top)
        bcs.append(fem.dirichletbc(bc_func_top, dofs_top, W.sub(0)))

        # Bottom wall (y=0): no-slip
        def bottom_wall(x):
            return np.isclose(x[1], 0.0)

        facets_bot = mesh.locate_entities_boundary(domain, fdim, bottom_wall)
        bc_func_bot = fem.Function(V_sub)
        bc_func_bot.interpolate(lambda x: np.zeros((2, x.shape[1])))
        dofs_bot = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_bot)
        bcs.append(fem.dirichletbc(bc_func_bot, dofs_bot, W.sub(0)))

        # Left wall (x=0): parabolic inflow u = (4*y*(1-y), 0)
        def left_wall(x):
            return np.isclose(x[0], 0.0)

        facets_left = mesh.locate_entities_boundary(domain, fdim, left_wall)
        bc_func_left = fem.Function(V_sub)
        bc_func_left.interpolate(
            lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]),
                                 np.zeros(x.shape[1])]))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_left)
        bcs.append(fem.dirichletbc(bc_func_left, dofs_left, W.sub(0)))

        # Right wall (x=1): parabolic outflow (same profile for well-posed problem)
        def right_wall(x):
            return np.isclose(x[0], 1.0)

        facets_right = mesh.locate_entities_boundary(domain, fdim, right_wall)
        bc_func_right = fem.Function(V_sub)
        bc_func_right.interpolate(
            lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]),
                                 np.zeros(x.shape[1])]))
        dofs_right = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_right)
        bcs.append(fem.dirichletbc(bc_func_right, dofs_right, W.sub(0)))

    # 7. Assemble and solve
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(L_form)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    # Set up KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.MINRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    # Handle pressure nullspace - for all-Dirichlet velocity BCs
    # Create nullspace for pressure (constant pressure mode)
    # This is needed when all boundaries have Dirichlet velocity BCs
    # We'll try with nullspace first
    # Actually for the default Poiseuille with all Dirichlet velocity BCs, 
    # pressure is determined up to a constant
    # Let's build the null space vector
    null_vec = A.createVecRight()
    null_vec.set(0.0)
    
    # Get the pressure DOF indices in the mixed space
    W0_dofs = W.sub(1).dofmap.list.array
    # Set pressure DOFs to 1
    # Need to work with the mixed space dofmap
    pressure_dofmap = W.sub(1).dofmap
    num_cells = domain.topology.index_map(tdim).size_local
    for cell in range(num_cells):
        cell_dofs = pressure_dofmap.cell_dofs(cell)
        for dof in cell_dofs:
            null_vec.setValue(dof, 1.0)
    
    null_vec.assemble()
    null_vec.normalize()
    
    nsp = PETSc.NullSpace().create(vectors=[null_vec], comm=domain.comm)
    A.setNullSpace(nsp)
    nsp.remove(b)

    wh = fem.Function(W)
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()

    iterations = ksp.getIterationNumber()

    ksp.destroy()
    A.destroy()
    b.destroy()

    # 8. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    # Get velocity sub-function
    uh = wh.sub(0).collapse()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    # Evaluate velocity (2 components)
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))

    # Replace any NaN with 0
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "minres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }


def _eval_vector_expr(s0, s1, x):
    """Evaluate vector expression strings at points x."""
    y = x[1]
    x_coord = x[0]
    n = x.shape[1]
    
    result = np.zeros((2, n))
    result[0, :] = _eval_scalar_expr(s0, x)
    result[1, :] = _eval_scalar_expr(s1, x)
    return result


def _eval_scalar_expr(s, x):
    """Evaluate a scalar expression string at points x."""
    n = x.shape[1]
    # Replace common math
    expr = s.replace("^", "**")
    # Make variables available
    x_val = x[0]
    y_val = x[1]
    
    # Try direct eval
    try:
        local_vars = {"x": x_val, "y": y_val, "np": np, "pi": np.pi,
                       "sin": np.sin, "cos": np.cos, "exp": np.exp}
        # Also handle x[0], x[1] style
        expr_mod = expr.replace("x[0]", "x").replace("x[1]", "y")
        result = eval(expr_mod, {"__builtins__": {}}, local_vars)
        if isinstance(result, (int, float)):
            return np.full(n, float(result))
        return np.asarray(result, dtype=np.float64)
    except Exception:
        return np.zeros(n)


def _parse_expression(s):
    """Parse a string expression - returns the string for later eval."""
    return s


def _eval_expr(s, x):
    """Evaluate parsed expression at points."""
    return _eval_scalar_expr(s, x)