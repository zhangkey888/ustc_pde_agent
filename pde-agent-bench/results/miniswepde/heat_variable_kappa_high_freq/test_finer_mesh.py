import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def compute_error(N=64, degree=1):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Parameters
    t_end = 0.1
    dt = 0.005
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution and coefficients
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t_end) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    kappa_expr = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
    
    # Source term
    u_exact = u_exact_expr
    du_dt = -u_exact
    grad_u = ufl.grad(u_exact)
    flux = kappa_expr * grad_u
    div_flux = ufl.div(flux)
    f_expr = du_dt - div_flux
    
    # Boundary condition
    el = V.element
    interpolation_points = el.interpolation_points
    u_bc_expr = fem.Expression(u_exact_expr, interpolation_points)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_bc_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Time-stepping
    u_n = fem.Function(V)
    u0_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u0_interp = fem.Expression(u0_expr, interpolation_points)
    u_n.interpolate(u0_interp)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = (1/dt_constant) * ufl.inner(u, v) * ufl.dx + ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (1/dt_constant) * ufl.inner(u_n, v) * ufl.dx + ufl.inner(f_expr, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    u_sol = fem.Function(V)
    
    # Solver
    ksp_solver = PETSc.KSP().create(domain.comm)
    ksp_solver.setOperators(A)
    ksp_solver.setType('preonly')
    ksp_solver.getPC().setType('lu')
    
    # Time loop
    t = 0.0
    steps = int(t_end / dt) + 1
    
    for step in range(steps):
        if t + dt > t_end:
            dt_constant.value = t_end - t
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp_solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        u_n.x.array[:] = u_sol.x.array
        t += dt_constant.value
    
    # Compute error in L2 norm
    error_expr = u_sol - u_exact_expr
    error_form = fem.form(ufl.inner(error_expr, error_expr) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    error_sq_global = comm.allreduce(error_sq, op=MPI.SUM)
    l2_error = np.sqrt(error_sq_global)
    
    if rank == 0:
        print(f"N={N}, degree={degree}: L2 error = {l2_error:.6e}")
    
    return l2_error

if __name__ == "__main__":
    # Test different configurations
    errors = []
    for N in [64, 96, 128]:
        err = compute_error(N=N, degree=1)
        errors.append((N, err))
    
    print("\nSummary:")
    for N, err in errors:
        print(f"N={N}: L2 error = {err:.6e}, meets requirement {err <= 1.22e-3}")
