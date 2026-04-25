import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Extract output specifications
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx = grid_spec.get("nx", 128)
    ny = grid_spec.get("ny", 128)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Mesh and function space (high resolution due to fast required time)
    mesh_res = 128
    element_degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # PDE Parameters
    epsilon = 0.01
    beta_vec = ufl.as_vector([12.0, 6.0])
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1]) + 0.3 * ufl.sin(12 * ufl.pi * x[0]) * ufl.sin(10 * ufl.pi * x[1])
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    v_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    # Peclet number
    Pe = v_norm * h / (2.0 * epsilon)
    # SUPG parameter tau
    tau = ufl.conditional(ufl.gt(Pe, 1.0), h / (2.0 * v_norm), h**2 / (12.0 * epsilon))
    
    # Residuals
    res_strong = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f
    v_supg = v + tau * ufl.dot(beta_vec, ufl.grad(v))
    
    # Weak form
    a = (epsilon * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx
    
    # SUPG addition
    a += tau * ufl.dot(beta_vec, ufl.grad(u)) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    a += tau * (-epsilon * ufl.div(ufl.grad(u))) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    L += tau * f * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    
    # Solve
    ksp_type = "preonly"
    pc_type = "lu"
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Interpolation to grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 100, "ny": 100, "bbox": [0,1,0,1]}}}
    res = solve(case)
    print("Solved. Output shape:", res["u"].shape)
