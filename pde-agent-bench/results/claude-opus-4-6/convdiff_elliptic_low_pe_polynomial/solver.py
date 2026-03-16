import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    epsilon = 0.3
    beta = [0.5, 0.3]
    
    # Mesh and element
    nx = ny = 64
    degree = 2
    
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    
    # Compute source term f = -eps * laplacian(u) + beta . grad(u)
    # u = x(1-x)y(1-y)
    # grad(u) = ((1-2x)y(1-y), x(1-x)(1-2y))
    # laplacian(u) = -2y(1-y) - 2x(1-x)
    # f = -eps * (-2y(1-y) - 2x(1-x)) + beta[0]*(1-2x)*y*(1-y) + beta[1]*x*(1-x)*(1-2y)
    f_expr = (-epsilon * (-2.0 * x[1] * (1 - x[1]) - 2.0 * x[0] * (1 - x[0]))
              + beta[0] * (1 - 2 * x[0]) * x[1] * (1 - x[1])
              + beta[1] * x[0] * (1 - x[0]) * (1 - 2 * x[1]))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Velocity vector
    b_vec = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                           fem.Constant(domain, ScalarType(beta[1]))])
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    # Bilinear and linear forms
    a = (eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.dot(b_vec, ufl.grad(u)), v) * ufl.dx)
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries (exact solution is zero on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="solve_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }