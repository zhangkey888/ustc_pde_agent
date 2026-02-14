import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve_adaptive(rel_tol=0.01):
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    total_iterations = 0
    
    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 2))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = fem.Constant(domain, ScalarType(1.0))
        x = ufl.SpatialCoordinate(domain)
        f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        f = fem.Expression(f_expr, V.element.interpolation_points)
        f_func = fem.Function(V)
        f_func.interpolate(f)
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        a_form = fem.form(a)
        L_form = fem.form(L)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        b = petsc.create_vector(L_form.function_spaces)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        u_sol = fem.Function(V)
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.GMRES)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        iterations = solver.getIterationNumber()
        total_iterations += iterations
        
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)
        norm_new = np.sqrt(norm_global)
        
        if norm_old is not None:
            relative_error = abs(norm_new - norm_old) / norm_new if norm_new > 0 else 1.0
            if relative_error < rel_tol:
                print(f'Converged at N={N} with relative error {relative_error}')
                return N, total_iterations, u_sol, domain
        norm_old = norm_new
    
    return 128, total_iterations, u_sol, domain

# Test with different tolerances
for tol in [0.01, 0.001]:
    print(f'\nTesting with relative tolerance {tol}')
    start = time.time()
    N, iters, u_sol, domain = solve_adaptive(rel_tol=tol)
    end = time.time()
    print(f'Final N={N}, iterations={iters}, time={end-start}')
