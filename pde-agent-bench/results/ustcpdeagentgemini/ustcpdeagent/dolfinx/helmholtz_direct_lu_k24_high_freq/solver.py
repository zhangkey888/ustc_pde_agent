import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 100)
    ny_out = grid_spec.get("ny", 100)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Sol param
    mesh_res = 120
    degree = 2
    k_val = 24.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term f = -div(grad(u_exact)) - k^2 * u_exact
    f = -ufl.div(ufl.grad(u_exact)) - (k_val**2) * u_exact
    
    # Trial/Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form and linear form
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_val**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Dirichlet BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Linear problem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_"
    )
    
    u_sol = problem.solve()
    
    # Evaluate on regular grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((len(pts),), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {
                "nx": 100, "ny": 100, "bbox": [0,1,0,1]
            }
        }
    }
    res = solve(spec)
    print("Shape:", res["u"].shape)
    print("Max val:", np.nanmax(res["u"]))
