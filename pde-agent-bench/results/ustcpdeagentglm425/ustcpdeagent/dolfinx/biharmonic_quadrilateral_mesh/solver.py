import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    out_grid = case_spec["output"]["grid"]
    nx = out_grid["nx"]
    ny = out_grid["ny"]
    bbox = out_grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Solver parameters - optimized for high accuracy within time budget
    mesh_res = 40
    element_degree = 4
    rtol = 1e-10
    ksp_type = "cg"
    pc_type = "hypre"
    
    # Create mesh on unit square with quadrilateral cells
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                    cell_type=mesh.CellType.quadrilateral)
    
    # Define exact solution and derived quantities
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    # v = -Delta u (Laplacian of u_exact)
    v_exact_ufl = 13*ufl.pi**2 * ufl.sin(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    # f = Delta^2 u = -Delta v (biharmonic source)
    f_ufl = 169*ufl.pi**4 * ufl.sin(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary setup
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Step 1: Solve -Delta v = f, with v = -Delta u_exact on boundary
    v_bc = fem.Function(V)
    v_bc.interpolate(fem.Expression(v_exact_ufl, V.element.interpolation_points))
    bc_v = fem.dirichletbc(v_bc, boundary_dofs)
    
    v = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
    
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))
    L1 = ufl.inner(f_func, w) * ufl.dx
    
    problem1 = petsc.LinearProblem(a1, L1, bcs=[bc_v],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
        petsc_options_prefix="bv_")
    
    t0 = time.perf_counter()
    v_sol = problem1.solve()
    v_sol.x.scatter_forward()
    its1 = problem1.solver.getIterationNumber()
    
    # Step 2: Solve -Delta u = v, with u = g on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    w2 = ufl.TestFunction(V)
    a2 = ufl.inner(ufl.grad(u), ufl.grad(w2)) * ufl.dx
    L2 = ufl.inner(v_sol, w2) * ufl.dx
    
    problem2 = petsc.LinearProblem(a2, L2, bcs=[bc_u],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
        petsc_options_prefix="bu_")
    
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    its2 = problem2.solver.getIterationNumber()
    t1 = time.perf_counter()
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx * ny, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
    
    u_values = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather across processes
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny, nx)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    M = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_local = fem.assemble_scalar(fem.form(M))
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_global:.6e}, solve time: {t1-t0:.4f}s, iterations: {its1+its2}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": its1 + its2,
        }
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {"time": False}
    }
    result = solve(case_spec)
