```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_res = 64
    degree = 2
    kappa = 10.0
    
    # Mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions
    # The exact solution is u = sin(pi*x)*sin(2*pi*y), which is 0 on all boundaries of [0,1]x[0,1]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # f = -div(kappa * grad(u)) = 10 * (pi^2 + 4*pi^2) * sin(pi*x)*sin(2*pi*y) = 50 * pi^2 * sin(pi*x)*sin(2*pi*y)
    f_expr = 50.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Linear problem setup
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="poisson_"
    )
    
    # Solve
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 uniform grid
    eval_nx, eval_ny = 50, 50
    x_coords = np.linspace(0, 1, eval_nx)
    y_coords = np.linspace(0, 1, eval_ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links