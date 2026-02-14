import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

def exact_solution(x):
    return np.sin(3*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1]))

def source_term(x):
    a = 3*np.pi*(x[0] + x[1])
    b = np.pi*(x[0] - x[1])
    return 20*np.pi*np.pi*np.sin(a)*np.sin(b)

resolutions = [32, 64, 128]
element_degrees = [1, 2]

for degree in element_degrees:
    print(f"\n=== Testing with element degree {degree} ===")
    for N in resolutions:
        start_time = time.time()
        
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        kappa = fem.Constant(domain, ScalarType(1.0))
        f_expr = fem.Function(V)
        f_expr.interpolate(lambda x: source_term(x))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        petsc_options = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 1000
        }
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options=petsc_options,
            petsc_options_prefix="test_"
        )
        u_sol = problem.solve()
        
        # Compute errors
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: exact_solution(x))
        
        error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
        l2_error = np.sqrt(fem.assemble_scalar(error_form))
        
        norm_exact_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
        norm_exact = np.sqrt(fem.assemble_scalar(norm_exact_form))
        relative_l2_error = l2_error / norm_exact
        
        # Also compute max pointwise error at mesh vertices
        mesh_points = domain.geometry.x
        u_sol_at_points = u_sol.eval(mesh_points, np.arange(len(mesh_points)))
        u_exact_at_points = exact_solution(mesh_points.T)
        max_error = np.max(np.abs(u_sol_at_points.flatten() - u_exact_at_points))
        
        elapsed = time.time() - start_time
        
        print(f"N={N}: L2 error={l2_error:.2e}, rel L2={relative_l2_error:.2e}, max error={max_error:.2e}, time={elapsed:.3f}s")
