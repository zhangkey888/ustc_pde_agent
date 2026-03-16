import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    nu_val = case_spec.get("pde", {}).get("viscosity", 0.9)
    source = case_spec.get("pde", {}).get("source_term", ["0.0", "0.0"])
    
    # Mesh resolution
    N = 64
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood elements: P2/P1
    degree_u = 2
    degree_p = 1
    
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(domain, mel)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term
    f = fem.Constant(domain, np.array([0.0, 0.0], dtype=ScalarType))
    nu = fem.Constant(domain, ScalarType(nu_val))
    
    # Bilinear form for Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    # Determine boundary type from case_spec
    # Case: stokes_no_exact_outflow_top_boundary
    # "no_exact" suggests we don't have an exact solution
    # "outflow_top_boundary" suggests outflow (natural BC) on top, Dirichlet elsewhere
    
    # We need to figure out the BCs from the case ID
    # For "outflow_top_boundary": top boundary has natural BC (outflow/do-nothing)
    # Other boundaries have Dirichlet (no-slip or specified)
    
    bc_specs = case_spec.get("pde", {}).get("boundary_conditions", [])
    
    bcs = []
    
    if bc_specs:
        for bc_spec in bc_specs:
            bc_type = bc_spec.get("type", "dirichlet")
            location = bc_spec.get("location", "")
            value = bc_spec.get("value", None)
            
            if bc_type.lower() == "neumann" or bc_type.lower() == "outflow" or bc_type.lower() == "natural":
                # Natural BC - do nothing
                continue
            
            # Dirichlet BC
            if "left" in location.lower():
                marker = lambda x: np.isclose(x[0], 0.0)
            elif "right" in location.lower():
                marker = lambda x: np.isclose(x[0], 1.0)
            elif "bottom" in location.lower():
                marker = lambda x: np.isclose(x[1], 0.0)
            elif "top" in location.lower():
                marker = lambda x: np.isclose(x[1], 1.0)
            elif "all" in location.lower() or "entire" in location.lower():
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                    np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            else:
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                    np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            
            # Parse value
            if value is not None:
                if isinstance(value, (list, tuple)):
                    val_strs = value
                else:
                    val_strs = [str(value), "0.0"]
                
                # Create interpolation function
                V_sub, sub_map = W.sub(0).collapse()
                u_bc = fem.Function(V_sub)
                
                # Build lambda for interpolation
                def make_bc_func(val_strs):
                    def bc_func(x):
                        result = np.zeros((domain.geometry.dim, x.shape[1]))
                        for i, vs in enumerate(val_strs):
                            if i < domain.geometry.dim:
                                # Evaluate expression
                                expr_str = str(vs)
                                # Replace common math
                                local_vars = {
                                    'x': x,
                                    'np': np,
                                    'pi': np.pi,
                                    'sin': np.sin,
                                    'cos': np.cos,
                                    'exp': np.exp,
                                }
                                # Try to handle x[0], x[1] references
                                try:
                                    result[i] = eval(expr_str, {"__builtins__": {}}, local_vars)
                                except:
                                    try:
                                        result[i] = float(expr_str) * np.ones(x.shape[1])
                                    except:
                                        result[i] = np.zeros(x.shape[1])
                        return result
                    return bc_func
                
                u_bc.interpolate(make_bc_func(val_strs))
                
                dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
                bcs.append(bc)
            else:
                # Zero Dirichlet
                V_sub, sub_map = W.sub(0).collapse()
                u_bc = fem.Function(V_sub)
                u_bc.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
                dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
                bcs.append(bc)
    else:
        # Default: no-slip on bottom, left, right; outflow on top
        # Based on case name "outflow_top_boundary"
        
        V_sub, sub_map = W.sub(0).collapse()
        
        # Bottom: no-slip
        bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        u_bottom = fem.Function(V_sub)
        u_bottom.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bottom_facets)
        bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))
        
        # Left: no-slip
        left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        u_left = fem.Function(V_sub)
        u_left.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, left_facets)
        bcs.append(fem.dirichletbc(u_left, dofs_left, W.sub(0)))
        
        # Right: no-slip
        right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
        u_right = fem.Function(V_sub)
        u_right.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_right = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, right_facets)
        bcs.append(fem.dirichletbc(u_right, dofs_right, W.sub(0)))
        
        # Top: outflow (natural BC) - do nothing
    
    # Pin pressure at one point to remove nullspace (since we might have all Dirichlet on velocity
    # except outflow on top, pressure should be determined, but let's be safe)
    # Actually with outflow BC, pressure is determined. But let's check if we need it.
    # For pure Dirichlet velocity BCs, we need pressure pinning. With outflow, we don't.
    # Since top is outflow, pressure should be OK.
    
    # Solve
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="stokes_"
    )
    
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract velocity
    u_sol = wh.sub(0).collapse()
    
    # Evaluate on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, gdim)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Replace any remaining NaN with 0 (boundary points that might not be found)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }