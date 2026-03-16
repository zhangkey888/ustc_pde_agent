import numpy as np
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("viscosity", 0.12))
    source = pde.get("source", ["0.0", "0.0"])
    
    # Also check oracle_config path
    if "oracle_config" in case_spec:
        oc = case_spec["oracle_config"]
        if "pde" in oc:
            pde2 = oc["pde"]
            nu_val = float(pde2.get("viscosity", nu_val))
            source = pde2.get("source", source)

    # Parse boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])
    if "oracle_config" in case_spec and "pde" in case_spec["oracle_config"]:
        bcs_spec = case_spec["oracle_config"]["pde"].get("boundary_conditions", bcs_spec)

    # 2. Create mesh
    N = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function space (Taylor-Hood P2/P1)
    degree_u = 2
    degree_p = 1
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)

    # 4. Define nonlinear residual
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    # Residual: viscous + convection + pressure gradient - source + incompressibility
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )

    # 5. Boundary conditions
    # Parse BC specs to determine what to apply
    # For channel inflow/outflow: typically parabolic inflow on left, do-nothing on right,
    # no-slip on top/bottom
    
    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def walls(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    bcs = []

    # Try to parse boundary conditions from spec
    has_custom_bcs = False
    inflow_expr = None
    noslip_markers = []
    outflow_type = None

    for bc_item in bcs_spec:
        bc_type = bc_item.get("type", "")
        location = bc_item.get("location", "")
        value = bc_item.get("value", None)

        if bc_type == "dirichlet" and "velocity" in bc_item.get("field", "velocity"):
            has_custom_bcs = True
            if "left" in location or "inflow" in location or "x=0" in location or location == "left":
                # Inflow BC
                if value is not None:
                    inflow_expr = value
                    # Apply inflow on left
                    facets_left = mesh.locate_entities_boundary(domain, fdim, left)
                    
                    V_sub, sub_map = W.sub(0).collapse()
                    u_inflow = fem.Function(V_sub)
                    
                    if isinstance(value, list) and len(value) == 2:
                        # Parse expressions
                        expr_x = str(value[0])
                        expr_y = str(value[1])
                        
                        def make_inflow_func(ex, ey):
                            def func(x):
                                result = np.zeros((2, x.shape[1]))
                                local_vars = {'x': x[0], 'y': x[1], 'np': np, 'pi': np.pi}
                                try:
                                    result[0] = eval(ex.replace('^', '**'), {"__builtins__": {}}, local_vars)
                                except:
                                    try:
                                        result[0] = float(ex) * np.ones_like(x[0])
                                    except:
                                        result[0] = np.zeros_like(x[0])
                                try:
                                    result[1] = eval(ey.replace('^', '**'), {"__builtins__": {}}, local_vars)
                                except:
                                    try:
                                        result[1] = float(ey) * np.ones_like(x[0])
                                    except:
                                        result[1] = np.zeros_like(x[0])
                                return result
                            return func
                        
                        u_inflow.interpolate(make_inflow_func(expr_x, expr_y))
                    elif isinstance(value, (int, float)):
                        u_inflow.interpolate(lambda x: np.full((2, x.shape[1]), float(value)))
                    
                    dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_left)
                    bcs.append(fem.dirichletbc(u_inflow, dofs_left, W.sub(0)))

            elif "top" in location or "bottom" in location or "wall" in location or "noslip" in location:
                noslip_markers.append(location)
                
                if "top" in location:
                    marker_func = top
                elif "bottom" in location:
                    marker_func = bottom
                elif "wall" in location:
                    marker_func = walls
                else:
                    marker_func = walls
                
                facets_wall = mesh.locate_entities_boundary(domain, fdim, marker_func)
                V_sub, sub_map = W.sub(0).collapse()
                u_noslip = fem.Function(V_sub)
                
                if value is not None and isinstance(value, list) and len(value) == 2:
                    vx, vy = float(value[0]), float(value[1])
                    u_noslip.interpolate(lambda x, vx=vx, vy=vy: np.vstack([
                        np.full_like(x[0], vx), np.full_like(x[0], vy)
                    ]))
                else:
                    u_noslip.interpolate(lambda x: np.zeros((2, x.shape[1])))
                
                dofs_wall = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_wall)
                bcs.append(fem.dirichletbc(u_noslip, dofs_wall, W.sub(0)))

        elif bc_type == "neumann" or bc_type == "do_nothing" or "outflow" in location:
            outflow_type = "do_nothing"

    # If no custom BCs parsed, apply default channel flow BCs
    if not has_custom_bcs:
        V_sub, sub_map = W.sub(0).collapse()
        
        # Parabolic inflow on left: u_x = 4*y*(1-y), u_y = 0
        u_inflow = fem.Function(V_sub)
        u_inflow.interpolate(lambda x: np.vstack([
            4.0 * x[1] * (1.0 - x[1]),
            np.zeros_like(x[0])
        ]))
        facets_left = mesh.locate_entities_boundary(domain, fdim, left)
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_left)
        bcs.append(fem.dirichletbc(u_inflow, dofs_left, W.sub(0)))

        # No-slip on top and bottom walls
        u_noslip = fem.Function(V_sub)
        u_noslip.interpolate(lambda x: np.zeros((2, x.shape[1])))
        facets_walls = mesh.locate_entities_boundary(domain, fdim, walls)
        dofs_walls = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_walls)
        bcs.append(fem.dirichletbc(u_noslip, dofs_walls, W.sub(0)))

        # Do-nothing (natural) BC on right boundary (outflow) - no explicit BC needed

    # 6. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Try to set MUMPS as the factor solver
    try:
        pc.setFactorSolverType("mumps")
    except:
        pass

    # Initial guess: solve Stokes first for better convergence
    # Set initial guess to zero (already is)
    w.x.array[:] = 0.0

    # Try Newton solve
    try:
        n_newton, converged = solver.solve(w)
    except RuntimeError:
        # If Newton fails, try with relaxation
        solver.relaxation_parameter = 0.5
        solver.max_it = 100
        w.x.array[:] = 0.0
        try:
            n_newton, converged = solver.solve(w)
        except RuntimeError:
            # Try even more relaxation
            solver.relaxation_parameter = 0.3
            solver.max_it = 200
            w.x.array[:] = 0.0
            n_newton, converged = solver.solve(w)

    w.x.scatter_forward()

    # 7. Extract velocity on evaluation grid
    nx_eval = 50
    ny_eval = 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Get velocity sub-function
    u_sol = w.sub(0).collapse()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    vel_magnitude = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        vel_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = vel_mag[idx]

    u_grid = vel_magnitude.reshape((nx_eval, ny_eval))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }