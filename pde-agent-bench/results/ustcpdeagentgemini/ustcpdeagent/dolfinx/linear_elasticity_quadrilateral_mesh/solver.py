import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import petsc4py.PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    
    # Adaptive parameters based on typical constraints
    res = 128
    deg = 2
    
    msh = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [res, res], cell_type=mesh.CellType.quadrilateral)
    
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", deg, (gdim,)))
    
    # Material parameters
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)
        
    x = ufl.SpatialCoordinate(msh)
    
    # Manufactured solution
    u_ex_x = ufl.sin(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    u_ex_y = ufl.sin(ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u_ex = ufl.as_vector([u_ex_x, u_ex_y])
    
    # Source term f = -div(sigma(u_ex))
    sigma_ex = sigma(u_ex)
    f = -ufl.div(sigma_ex)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Dirichlet BC on entire boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    expr = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = LinearProblem(a, L, bcs=[bc], 
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, 
                            petsc_options_prefix="elast_")
    u_sol = problem.solve()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals_sampled = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals_sampled[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals_sampled, axis=1).reshape(ny_out, nx_out)
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": res,
            "element_degree": deg,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
