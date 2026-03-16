```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Time parameters
    t_end = 0.12
    dt = 0.03
    n_steps = int(round(t_end / dt))
    
    # Spatial discretization
    nx, ny = 64, 64
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [nx, ny], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Initial condition (u0 = 0.0)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    # Variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f_const = fem.Constant(domain, PETSc.ScalarType(1.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Weak form: ∫ u*v dx + dt * ∫ κ ∇u·∇v dx = ∫ u_n*v dx + dt * ∫ f*v dx
    a = u * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_const * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Linear solver setup
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t = 0.0
    total_iterations = 0
    
    # Time-stepping loop
    for i in range(n_steps):
        t += dt
        
        # Assemble RHS vector
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply boundary conditions to RHS
        pet