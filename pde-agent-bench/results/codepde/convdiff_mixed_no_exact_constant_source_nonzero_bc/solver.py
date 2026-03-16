import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec["pde"]
    epsilon = pde_config["epsilon"]
    beta = pde_config["beta"]
    source_term_val = pde_config.get("source_term", 1.0)
    
    # Get boundary conditions
    bcs_spec = pde_config.get("boundary_conditions", {})
    
    # High Peclet number -> need SUPG stabilization and fine mesh
    # Pe ~ 2400, so we need good stabilization
    nx, ny = 128, 128
    degree = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Parameters
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    beta_vec = fem.Constant(domain, PETSc.ScalarType(np.array(beta, dtype=np.float64)))
    f = fem.Constant(domain, PETSc.ScalarType(source_term_val))
    
    # Element size
    h = ufl.CellDiameter(domain)
    
    # Peclet number based on element
    beta_mag = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_h = beta_mag * h / (2.0 * eps_c)
    
    # SUPG stabilization parameter (optimal formula)
    # tau = h / (2 * |beta|) * (coth(Pe_h) - 1/Pe_h)
    # Use approximation: for large Pe_h, tau ~ h/(2*|beta|)
    # More robust: tau = h^2 / (4*eps) when Pe_h < 1, else h/(2*|beta|)
    # Use the standard formula with a smooth approximation
    tau = h / (2.0 * beta_mag + 1e-10) * (1.0 - 1.0 / (Pe_h + 1e-10))
    # Clamp tau to be non-negative using conditional
    tau = ufl.conditional(ufl.gt(Pe_h, 1.0), h / (2.0 * beta_mag + 1e-10), h * h / (12.0 * eps_c + 1e-10))
    
    # Standard Galerkin terms
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_gal = f * v * ufl.dx
    
    # SUPG additional terms
    # Residual of strong form applied to trial function: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within each element
    # So residual ~ beta.grad(u) - f
    R_u = ufl.dot(beta_vec, ufl.grad(u))  # -eps * div(grad(u)) is zero for P1
    R_f = f
    
    # SUPG test function modification
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(R_u, v_supg) * ufl.dx
    L_supg = ufl.inner(R_f, v_supg) * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Parse boundary conditions from spec
    # Default: try to get from bcs_spec
    bc_list = []
    
    if bcs_spec:
        # Handle different boundary condition formats
        if "dirichlet" in bcs_spec:
            dirichlet_bcs = bcs_spec["dirichlet"]
            if isinstance(dirichlet_bcs, dict):
                dirichlet_bcs = [dirichlet_bcs]
            for dbc in dirichlet_bcs:
                location = dbc.get("location", "all")
                value = dbc.get("value", 0.0)
                
                if location == "all":
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                elif location == "left":
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[0], 0.0))
                elif location == "right":
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[0], 1.0))
                elif location == "bottom":
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[1], 0.0))
                elif location == "top":
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[1], 1.0))
                else:
                    boundary_facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                
                dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
                
                if callable(value):
                    u_bc = fem.Function(V)
                    u_bc.interpolate(value)
                    bc_list.append(fem.dirichletbc(u_bc, dofs))
                elif isinstance(value, (int, float)):
                    bc_list.append(fem.dirichletbc(PETSc.ScalarType(value), dofs, V))
                else:
                    bc_list.append(fem.dirichletbc(PETSc.ScalarType(float(value)), dofs, V))
        else:
            # Try to interpret bcs_spec directly
            # Fallback: apply value on all boundaries
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            
            value = bcs_spec.get("value", 0.0)
            if isinstance(value, str):
                # Parse expression
                u_bc_func = fem.Function(V)
                u_bc_func.interpolate(lambda x: np.full(x.shape[1], 0.0))
                bc_list.append(fem.dirichletbc(u_bc_func, dofs))
            else:
                bc_list.append(fem.dirichletbc(PETSc.ScalarType(float(value)), dofs, V))
    else:
        # Default: zero Dirichlet on all boundaries
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc_list.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
    
    # Handle nonzero_bc case - check if there's specific BC info
    if "nonzero_bc" in str(case_spec.get("case_id", "")):
        # Check for specific BC values in the spec
        pass  # Already handled above
    
    # Parse boundary conditions more carefully
    if "boundary_conditions" in pde_config:
        bc_info = pde_config["boundary_conditions"]
        bc_list = []  # Reset
        
        if isinstance(bc_info, list):
            for bc_item in bc_info:
                loc = bc_item.get("location", "all")
                val = bc_item.get("value", 0.0)
                bc_type = bc_item.get("type", "dirichlet")
                
                if bc_type != "dirichlet":
                    continue
                
                if loc == "all":
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                elif loc == "left":
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[0], 0.0))
                elif loc == "right":
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[0], 1.0))
                elif loc == "bottom":
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[1], 0.0))
                elif loc == "top":
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.isclose(x[1], 1.0))
                else:
                    facets = mesh.locate_entities_boundary(
                        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                
                dofs = fem.locate_dofs_topological(V, fdim, facets)
                
                if isinstance(val, (int, float)):
                    bc_list.append(fem.dirichletbc(PETSc.ScalarType(float(val)), dofs, V))
                elif isinstance(val, str):
                    u_bc_f = fem.Function(V)
                    try:
                        # Try to evaluate as expression
                        u_bc_f.interpolate(lambda x, expr=val: np.full(x.shape[1], float(eval(expr))))
                    except:
                        u_bc_f.interpolate(lambda x: np.zeros(x.shape[1]))
                    bc_list.append(fem.dirichletbc(u_bc_f, dofs))
                    
        elif isinstance(bc_info, dict):
            # Could be {"dirichlet": [...]} or direct specification
            if "dirichlet" in bc_info:
                d_list = bc_info["dirichlet"]
                if isinstance(d_list, dict):
                    d_list = [d_list]
                for bc_item in d_list:
                    loc = bc_item.get("location", "all")
                    val = bc_item.get("value", 0.0)
                    
                    if loc == "all":
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                    elif loc == "left":
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.isclose(x[0], 0.0))
                    elif loc == "right":
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.isclose(x[0], 1.0))
                    elif loc == "bottom":
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.isclose(x[1], 0.0))
                    elif loc == "top":
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.isclose(x[1], 1.0))
                    else:
                        facets = mesh.locate_entities_boundary(
                            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                    
                    dofs = fem.locate_dofs_topological(V, fdim, facets)
                    
                    if isinstance(val, (int, float)):
                        bc_list.append(fem.dirichletbc(PETSc.ScalarType(float(val)), dofs, V))
                    elif isinstance(val, str):
                        u_bc_f = fem.Function(V)
                        try:
                            fval = float(val)
                            u_bc_f.interpolate(lambda x, fv=fval: np.full(x.shape[1], fv))
                        except:
                            u_bc_f.interpolate(lambda x: np.zeros(x.shape[1]))
                        bc_list.append(fem.dirichletbc(u_bc_f, dofs))
            else:
                # Direct value specification
                val = bc_info.get("value", 0.0)
                facets = mesh.locate_entities_boundary(
                    domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                dofs = fem.locate_dofs_topological(V, fdim, facets)
                if isinstance(val, (int, float)):
                    bc_list.append(fem.dirichletbc(PETSc.ScalarType(float(val)), dofs, V))
    
    if not bc_list:
        # Fallback
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc_list.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bc_list,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }