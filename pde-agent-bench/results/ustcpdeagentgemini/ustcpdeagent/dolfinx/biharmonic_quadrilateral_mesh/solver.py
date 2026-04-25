import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    out_grid = case_spec.get("output", {}).get("grid", {})
    nx_out = out_grid.get("nx", 50)
    ny_out = out_grid.get("ny", 50)
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    nx_mesh = 128
    ny_mesh = 128
    element_degree = 3
    
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [nx_mesh, ny_mesh], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    
    v_ex = -(u_ex.dx(0).dx(0) + u_ex.dx(1).dx(1))
    f = -(v_ex.dx(0).dx(0) + v_ex.dx(1).dx(1))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    v_bc_func = fem.Function(V)
    v_expr = fem.Expression(v_ex, V.element.interpolation_points)
    v_bc_func.interpolate(v_expr)
    bc_v = fem.dirichletbc(v_bc_func, dofs)
    
    v = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    a_v = ufl.inner(ufl.grad(v), ufl.grad(phi)) * ufl.dx
    L_v = ufl.inner(f, phi) * ufl.dx
    
    problem_v = petsc.LinearProblem(a_v, L_v, bcs=[bc_v], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="biharmonic_v_")
    v_sol = problem_v.solve()
    
    u_bc_func = fem.Function(V)
    u_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs)
    
    u = ufl.TrialFunction(V)
    psi = ufl.TestFunction(V)
    a_u = ufl.inner(ufl.grad(u), ufl.grad(psi)) * ufl.dx
    L_u = ufl.inner(v_sol, psi) * ufl.dx
    
    problem_u = petsc.LinearProblem(a_u, L_u, bcs=[bc_u], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="biharmonic_u_")
    u_sol = problem_u.solve()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((len(points),), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 2
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(spec)
    print("Shape:", res["u"].shape)
    print("Max val:", np.nanmax(res["u"]))
