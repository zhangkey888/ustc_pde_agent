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
    nx, ny = 128, 128
    degree = 2
    k = 16.0
    
    # Create quadrilateral mesh
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [nx, ny], cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (u = 0 on ∂Ω)
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
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.cos(5 * ufl.pi * x[1])
    
    # Helmholtz equation: -∇²u - k²u = f
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Linear solver
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    try:
        iterations = problem.solver.getIterationNumber()
    except:
        iterations = 1
        
    # Evaluate on 50x50 grid
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim