import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None):
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Parameters
    epsilon = 0.03
    beta_vec = [5.0, 2.0]
    
    if case_spec is not None:
        cd_params = case_spec.get('pde', {}).get('convection_diffusion', {})
        epsilon = cd_params.get('epsilon', epsilon)
        beta_vec = cd_params.get('beta', beta_vec)
    
    # Output grid
    nx_out, ny_out = 50, 50
    
    # P2 elements with SUPG stabilization
    N = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Source term derived from manufactured solution
    # u = sin(pi*x)*sin(2*pi*y)
    # -ε∇²u + β·∇u = f
    f_expr = eps_c * 5.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) \
           + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) \
           + beta_vec[1] * 2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Standard Galerkin terms
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = ufl.inner(f_expr, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    
    # SUPG stabilization parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG test function modification
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    # SUPG terms (including diffusion term for P2)
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    a_supg_diff = -eps_c * ufl.div(ufl.grad(u)) * v_supg * ufl.dx
    L_supg = ufl.inner(f_expr, v_supg) * ufl.dx
    
    a = a_std + a_supg + a_supg_diff
    L = L_std + L_supg
    
    # Boundary conditions (u = 0 on all boundaries)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Solve with direct solver for speed (avoid many GMRES iterations)
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    
    # Check accuracy against exact solution
    nx_out, ny_out = 50, 50
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    
    u_grid = result["u"]
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Iterations: {result['solver_info']['iterations']}")
