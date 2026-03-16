import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Extract viscosity
    nu_val = float(pde.get("viscosity", 0.9))
    
    # Extract source term
    source = pde.get("source_term", ["0.0", "0.0"])
    if isinstance(source, str):
        source = [source, source]
    
    # Extract boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Taylor-Hood mixed elements: P2/P1
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    TH = P2 * P1
    
    W = fem.functionspace(domain, TH)
    
    # Also create individual spaces for interpolation of BCs
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    
    # 4. Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    nu = fem.Constant(domain, default_scalar_type(nu_val))
    
    # Source term
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
    
    # Helper to create velocity BC functions
    def parse_bc_value(bc_info):
        """Parse boundary condition value from spec."""
        bc_type = bc_info.get("type", "dirichlet")
        value = bc_info.get("value", None)
        
        if value is None:
            return lambda x: np.zeros((2, x.shape[1]))
        
        if isinstance(value, list):
            # Vector value for velocity
            v0_str = str(value[0])
            v1_str = str(value[1])
            
            def bc_func(x_coord):
                result = np.zeros((2, x_coord.shape[1]))
                # Evaluate expressions
                local_vars = {'x': x_coord, 'np': np, 'pi': np.pi}
                # Replace x[0], x[1] with actual arrays
                expr0 = v0_str.replace('x[0]', 'x_coord[0]').replace('x[1]', 'x_coord[1]')
                expr1 = v1_str.replace('x[0]', 'x_coord[0]').replace('x[1]', 'x_coord[1]')
                
                try:
                    r0 = eval(expr0, {"__builtins__": {}}, {'x_coord': x_coord, 'np': np, 'pi': np.pi, 'sin': np.sin, 'cos': np.cos})
                    if np.isscalar(r0):
                        result[0, :] = r0
                    else:
                        result[0, :] = r0
                except:
                    result[0, :] = float(v0_str) if v0_str.replace('.', '').replace('-', '').replace('e', '').isdigit() else 0.0
                
                try:
                    r1 = eval(expr1, {"__builtins__": {}}, {'x_coord': x_coord, 'np': np, 'pi': np.pi, 'sin': np.sin, 'cos': np.cos})
                    if np.isscalar(r1):
                        result[1, :] = r1
                    else:
                        result[1, :] = r1
                except:
                    result[1, :] = float(v1_str) if v1_str.replace('.', '').replace('-', '').replace('e', '').isdigit() else 0.0
                
                return result
            
            return bc_func
        else:
            val = float(value)
            return lambda x: np.full((2, x.shape[1]), val)
    
    def safe_eval_scalar(expr_str, x_coord):
        """Safely evaluate a scalar expression string."""
        expr_str = str(expr_str).strip()
        
        # Try direct float conversion first
        try:
            val = float(expr_str)
            return np.full(x_coord.shape[1], val)
        except ValueError:
            pass
        
        # Try symbolic evaluation
        local_ns = {
            'x': x_coord,
            'np': np,
            'pi': np.pi,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'abs': np.abs,
        }
        
        modified = expr_str.replace('x[0]', 'x[0]').replace('x[1]', 'x[1]')
        
        try:
            result = eval(modified, {"__builtins__": {}}, local_ns)
            if np.isscalar(result):
                return np.full(x_coord.shape[1], float(result))
            return np.asarray(result)
        except:
            return np.zeros(x_coord.shape[1])
    
    def safe_eval_vector(value_list, x_coord):
        """Evaluate a vector BC value."""
        result = np.zeros((2, x_coord.shape[1]))
        for i, expr_str in enumerate(value_list[:2]):
            result[i, :] = safe_eval_scalar(expr_str, x_coord)
        return result
    
    # Process boundary conditions
    domain.topology.create_connectivity(fdim, tdim)
    
    for bc_info in bcs_spec:
        bc_type = bc_info.get("type", "dirichlet")
        location = bc_info.get("location", "")
        value = bc_info.get("value", None)
        
        if bc_type == "dirichlet":
            # Determine which boundary
            if location == "left" or location == "x=0":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif location == "right" or location == "x=1":
                marker = lambda x: np.isclose(x[0], 1.0)
            elif location == "bottom" or location == "y=0":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif location == "top" or location == "y=1":
                marker = lambda x: np.isclose(x[1], 1.0)
            elif location == "all" or location == "boundary":
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            else:
                # Try to parse more complex location strings
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            
            # Create velocity BC
            if value is not None:
                u_bc = fem.Function(V)
                if isinstance(value, list):
                    u_bc.interpolate(lambda x, val=value: safe_eval_vector(val, x))
                elif isinstance(value, (int, float)):
                    u_bc.interpolate(lambda x, val=value: np.full((2, x.shape[1]), float(val)))
                else:
                    try:
                        fval = float(value)
                        u_bc.interpolate(lambda x, val=fval: np.full((2, x.shape[1]), val))
                    except:
                        u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
            else:
                u_bc = fem.Function(V)
                u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
            
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
        
        elif bc_type == "outflow" or bc_type == "neumann" or bc_type == "do_nothing":
            # Natural BC - nothing to do (it's built into the weak form)
            pass
    
    # If no BCs were specified, apply zero velocity on all boundaries
    if len(bcs) == 0:
        all_facets = mesh.locate_entities_boundary(
            domain, fdim,
            lambda x: np.ones(x.shape[1], dtype=bool)
        )
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, all_facets)
        bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
        bcs.append(bc)
    
    # 6. Solve using assembled system with pressure nullspace handling
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.assemble_vector(L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Create solution function
    w_sol = fem.Function(W)
    
    # Setup KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.MINRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Try MUMPS if available
    try:
        pc.setFactorSolverType("mumps")
    except:
        try:
            pc.setFactorSolverType("superlu_dist")
        except:
            ksp.setType(PETSc.KSP.Type.GMRES)
            pc.setType(PETSc.PC.Type.ILU)
    
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    ksp.setUp()
    
    ksp.solve(b, w_sol.x.petsc_vec)
    w_sol.x.scatter_forward()
    
    iterations = ksp.getIterationNumber()
    ksp_type_used = ksp.getType()
    pc_type_used = pc.getType()
    
    # 7. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Extract velocity sub-function
    u_sub = w_sol.sub(0).collapse()
    
    # Point evaluation
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
    
    # Velocity magnitude
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sub.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Clean up
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }