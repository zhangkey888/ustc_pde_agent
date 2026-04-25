import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.05
    beta_val = np.array([3.0, 3.0])
    
    N = 96
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    
    # f = -eps*lap(u) + beta . grad(u)
    grad_ue = ufl.grad(u_exact)
    lap_ue = ufl.div(grad_ue)
    f = -eps_c * lap_ue + ufl.dot(beta, grad_ue)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    b_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = b_norm * h / (2 * eps_c)
    # tau for high Pe
    tau = h / (2 * b_norm) * (1.0 / ufl.tanh(Pe_h) - 1.0/Pe_h)
    
    # Residual for SUPG
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a_supg = tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a += a_supg
    L += L_supg
    
    # BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()
    
    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[0],), np.nan)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny, nx)
    
    # Accuracy check
    u_ex_grid = np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid)**2))
    print(f"RMS error vs exact: {err:.3e}, iters: {iters}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": iters,
        }
    }

if __name__ == "__main__":
    import time
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    res = solve(case_spec)
    print(f"Time: {time.time()-t0:.3f}s")
