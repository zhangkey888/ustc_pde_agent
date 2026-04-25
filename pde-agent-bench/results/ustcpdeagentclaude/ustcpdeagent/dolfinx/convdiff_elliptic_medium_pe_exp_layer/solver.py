import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    eps_val = 0.05
    beta_val = np.array([4.0, 0.0])
    
    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    # Manufactured solution
    u_exact = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # f = -eps * laplacian(u) + beta . grad(u)
    # lap = 4*exp(2x)*sin(pi*y) - pi^2 * exp(2x)*sin(pi*y) = (4-pi^2) * u_exact
    # beta . grad(u) = 4 * 2 * exp(2x)*sin(pi*y) = 8 * u_exact
    # f = -eps*(4-pi^2)*u + 8*u
    lap_u = 4.0 * ufl.exp(2.0*x[0])*ufl.sin(ufl.pi*x[1]) - (ufl.pi**2) * ufl.exp(2.0*x[0])*ufl.sin(ufl.pi*x[1])
    grad_u = ufl.as_vector([2.0*ufl.exp(2.0*x[0])*ufl.sin(ufl.pi*x[1]),
                             ufl.pi*ufl.exp(2.0*x[0])*ufl.cos(ufl.pi*x[1])])
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    f = -eps_c * lap_u + ufl.dot(beta, grad_u)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0/Pe_h)
    
    # strong residual
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_L = f
    
    a += tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(r_L, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    its = problem.solver.getIterationNumber()
    
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
    
    points_on_proc = []
    cells = []
    emap = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            emap.append(i)
    
    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[emap] = vals.flatten()
    
    u_grid = u_values.reshape(ny, nx)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    
    # Check error
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(2*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((result["u"] - u_ex)**2))
    print(f"L2 error on grid: {err:.3e}")
    print(f"Max error: {np.max(np.abs(result['u'] - u_ex)):.3e}")
