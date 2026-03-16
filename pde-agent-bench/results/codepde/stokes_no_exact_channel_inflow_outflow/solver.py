import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from dolfinx import nls
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    nu_val = float(pde.get("viscosity", 1.0))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Taylor-Hood mixed elements: P2/P1
    P2 = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    P1 = fem.functionspace(domain, ("Lagrange", 1))
    
    # Mixed function space via MixedElement approach
    vel_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    pres_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    mixed_elem = basix.ufl.mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.as_vector([0.0, 0.0])
    
    # Bilinear form: Stokes
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions
    # Parse boundary conditions from case_spec
    bcs = []
    
    # We need to figure out what BCs are specified
    # For channel inflow/outflow on unit square:
    # Typical: parabolic inflow on left, do-nothing on right, no-slip on top/bottom
    
    V_sub, _ = W.sub(0).collapse()
    
    def parse_bc_value(bc_info, space):
        """Parse boundary condition value and return a fem.Function."""
        bc_type = bc_info.get("type", "dirichlet")
        value = bc_info.get("value", None)
        
        u_bc = fem.Function(space)
        
        if value is not None:
            if isinstance(value, list):
                # Vector BC
                val_strs = [str(v) for v in value]
                
                def bc_func(x):
                    result = np.zeros((domain.geometry.dim, x.shape[1]))
                    for i, vs in enumerate(val_strs):
                        vs_clean = vs.strip()
                        if 'x[0]' in vs_clean or 'x[1]' in vs_clean or 'x' in vs_clean:
                            # Expression involving coordinates
                            local_vars = {'x': x, 'np': np, 'pi': np.pi}
                            # Replace x[0], x[1] with x[0], x[1]
                            expr = vs_clean.replace('x[0]', 'x[0]').replace('x[1]', 'x[1]')
                            result[i] = eval(expr, {"__builtins__": {}}, local_vars)
                        else:
                            try:
                                result[i] = float(vs_clean) * np.ones(x.shape[1])
                            except:
                                local_vars = {'x': x, 'np': np, 'pi': np.pi}
                                result[i] = eval(vs_clean, {"__builtins__": {}}, local_vars) * np.ones(x.shape[1])
                    return result
                
                u_bc.interpolate(bc_func)
            else:
                val_str = str(value).strip()
                try:
                    val_float = float(val_str)
                    u_bc.interpolate(lambda x: np.full((domain.geometry.dim, x.shape[1]), val_float))
                except:
                    pass
        else:
            u_bc.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        
        return u_bc
    
    has_explicit_bcs = len(bcs_spec) > 0
    
    if has_explicit_bcs:
        for bc_info in bcs_spec:
            bc_type = bc_info.get("type", "dirichlet")
            location = bc_info.get("location", "")
            
            if bc_type.lower() == "neumann" or bc_type.lower() == "do_nothing":
                # Natural BC, skip
                continue
            
            # Determine boundary marker
            if "left" in location.lower() or "x=0" in location.lower() or "x ==0" in location.lower() or "x == 0" in location.lower():
                marker = lambda x: np.isclose(x[0], 0.0)
            elif "right" in location.lower() or "x=1" in location.lower() or "x ==1" in location.lower() or "x == 1" in location.lower():
                marker = lambda x: np.isclose(x[0], 1.0)
            elif "bottom" in location.lower() or "y=0" in location.lower() or "y ==0" in location.lower() or "y == 0" in location.lower():
                marker = lambda x: np.isclose(x[1], 0.0)
            elif "top" in location.lower() or "y=1" in location.lower() or "y ==1" in location.lower() or "y == 1" in location.lower():
                marker = lambda x: np.isclose(x[1], 1.0)
            elif "all" in location.lower() or "boundary" in location.lower():
                marker = lambda x: np.ones(x.shape[1], dtype=bool)
            else:
                # Try to detect from expression
                marker = lambda x: np.ones(x.shape[1], dtype=bool)
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            
            # Check if this is a velocity BC or component BC
            component = bc_info.get("component", None)
            
            if component is not None:
                # Single component BC
                comp_idx = int(component)
                W_sub_comp, sub_comp_map = W.sub(0).sub(comp_idx).collapse()
                u_bc = fem.Function(W_sub_comp)
                value = bc_info.get("value", "0.0")
                val_str = str(value).strip()
                
                def make_scalar_interp(val_str_local):
                    def interp_func(x):
                        try:
                            return np.full(x.shape[1], float(val_str_local))
                        except:
                            local_vars = {'x': x, 'np': np, 'pi': np.pi}
                            return eval(val_str_local, {"__builtins__": {}}, local_vars) * np.ones(x.shape[1])
                    return interp_func
                
                u_bc.interpolate(make_scalar_interp(val_str))
                dofs = fem.locate_dofs_topological((W.sub(0).sub(comp_idx), W_sub_comp), fdim, facets)
                bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0).sub(comp_idx)))
            else:
                # Full velocity BC
                u_bc = parse_bc_value(bc_info, V_sub)
                dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    else:
        # Default: channel flow with parabolic inflow on left, do-nothing on right, no-slip on top/bottom
        
        # No-slip on top and bottom walls
        def top_wall(x):
            return np.isclose(x[1], 1.0)
        
        def bottom_wall(x):
            return np.isclose(x[1], 0.0)
        
        # Inflow on left
        def left_wall(x):
            return np.isclose(x[0], 0.0)
        
        # No-slip top
        facets_top = mesh.locate_entities_boundary(domain, fdim, top_wall)
        u_noslip = fem.Function(V_sub)
        u_noslip.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_top)
        bcs.append(fem.dirichletbc(u_noslip, dofs_top, W.sub(0)))
        
        # No-slip bottom
        facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom_wall)
        u_noslip2 = fem.Function(V_sub)
        u_noslip2.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_bottom)
        bcs.append(fem.dirichletbc(u_noslip2, dofs_bottom, W.sub(0)))
        
        # Parabolic inflow on left: u_x = 4*y*(1-y), u_y = 0
        facets_left = mesh.locate_entities_boundary(domain, fdim, left_wall)
        u_inflow = fem.Function(V_sub)
        u_inflow.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1])]))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_left)
        bcs.append(fem.dirichletbc(u_inflow, dofs_left, W.sub(0)))
        
        # Right boundary: do-nothing (natural BC) - no Dirichlet needed
    
    # 6. Solve
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
    
    # Create solution function
    wh = fem.Function(W)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.MINRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    
    # Try MUMPS direct solver for robustness
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc.setType(PETSc.PC.Type.LU)
    # Try to use MUMPS if available
    try:
        pc.setFactorSolverType("mumps")
    except:
        pass
    
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=2000)
    solver.setUp()
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    ksp_type = "preonly"
    pc_type = "lu"
    
    # 7. Extract velocity on evaluation grid
    nx_eval = 100
    ny_eval = 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    # Get velocity subfunction
    uh = wh.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    vel_magnitude = np.full(nx_eval * ny_eval, np.nan)
    
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_array, cells_array)
        # vals shape: (n_points, gdim)
        if vals.ndim == 1:
            vel_magnitude[eval_map] = np.abs(vals)
        else:
            # Compute velocity magnitude
            vel_mag = np.sqrt(np.sum(vals**2, axis=1))
            vel_magnitude[eval_map] = vel_mag
    
    u_grid = vel_magnitude.reshape((nx_eval, ny_eval))
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }