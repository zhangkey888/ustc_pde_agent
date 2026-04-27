import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    nu_val = float(case_spec["pde"]["coefficients"]["nu"])
    bc_specs = case_spec["pde"]["boundary_conditions"]
    
    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh resolution - use high resolution for accuracy
    N = 128
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    
    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Viscosity
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Source term - parse from case_spec
    src = case_spec["pde"]["source_term"]
    if isinstance(src, list):
        f_vals = [float(s) for s in src]
        f = fem.Constant(msh, PETSc.ScalarType(tuple(f_vals)))
    else:
        f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Bilinear form for Stokes: -nu*laplacian(u) + grad(p) = f, div(u) = 0
    # Weak form: nu*inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx = inner(f,v)*dx
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    bcs = []
    
    bc_locations_set = set()
    
    for bc_spec in bc_specs:
        bc_type = bc_spec["type"]
        location = bc_spec["location"]
        value = bc_spec["value"]
        bc_locations_set.add(location)
        
        if bc_type == "dirichlet":
            if location == "x0":
                marker = lambda x: np.isclose(x[0], bbox[0])
            elif location == "x1":
                marker = lambda x: np.isclose(x[0], bbox[1])
            elif location == "y0":
                marker = lambda x: np.isclose(x[1], bbox[2])
            elif location == "y1":
                marker = lambda x: np.isclose(x[1], bbox[3])
            else:
                continue
            
            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            
            u_bc = fem.Function(V)
            
            if isinstance(value, list) and len(value) == 2:
                expr_x_str = value[0]
                expr_y_str = value[1]
                
                def make_bc_func(ex, ey):
                    def bc_func(x):
                        from numpy import sin, cos, pi, exp
                        X = x[0]
                        y = x[1]
                        local_vars = {"sin": sin, "cos": cos, "pi": pi, "exp": exp, 
                                      "x": X, "y": y, "np": np}
                        vx = eval(ex, {"__builtins__": {}}, local_vars)
                        vy = eval(ey, {"__builtins__": {}}, local_vars)
                        vx = vx * np.ones_like(X) if np.isscalar(vx) else vx
                        vy = vy * np.ones_like(X) if np.isscalar(vy) else vy
                        return np.vstack([vx, vy])
                    return bc_func
                
                u_bc.interpolate(make_bc_func(expr_x_str, expr_y_str))
            
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    
    # Check if all boundaries have Dirichlet velocity BCs -> need pressure pinning
    all_dirichlet = {"x0", "x1", "y0", "y1"}.issubset(bc_locations_set)
    
    if all_dirichlet:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2])
        )
        if len(p_dofs[0]) > 0:
            p0 = fem.Function(Q)
            p0.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
            bcs.append(bc_p)
    
    # Solve using MUMPS direct solver (handles saddle-point systems well)
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((len(pts), gdim), 0.0)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }
    
    return {"u": u_grid, "solver_info": solver_info}
