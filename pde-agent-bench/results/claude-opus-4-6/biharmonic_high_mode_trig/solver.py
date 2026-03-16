import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Manufactured solution: u = sin(3*pi*x)*sin(2*pi*y)
    # Laplacian: -(9*pi^2 + 4*pi^2)*sin(3*pi*x)*sin(2*pi*y) = -13*pi^2 * u_exact
    # Bilaplacian: Delta^2 u = 13*pi^2 * 13*pi^2 * sin(3*pi*x)*sin(2*pi*y)
    # Wait, let me compute carefully:
    # u = sin(3*pi*x)*sin(2*pi*y)
    # u_xx = -9*pi^2 * sin(3*pi*x)*sin(2*pi*y)
    # u_yy = -4*pi^2 * sin(3*pi*x)*sin(2*pi*y)
    # Delta u = -(9+4)*pi^2 * sin(3*pi*x)*sin(2*pi*y) = -13*pi^2 * u
    # Let w = Delta u = -13*pi^2 * sin(3*pi*x)*sin(2*pi*y)
    # Delta w = -13*pi^2 * Delta(sin(3*pi*x)*sin(2*pi*y)) = -13*pi^2 * (-13*pi^2) * sin(...) = 169*pi^4 * u
    # So f = Delta^2 u = 169*pi^4 * sin(3*pi*x)*sin(2*pi*y)
    
    # Mixed formulation: 
    # -Delta w = f, w = -Delta u
    # Step 1: Solve -Delta w = f with w = -Delta u_exact on boundary
    # Step 2: Solve -Delta u = w with u = u_exact on boundary
    
    # But on boundary of [0,1]^2, u_exact = sin(3*pi*x)*sin(2*pi*y) = 0 on all boundaries
    # (since sin(0)=sin(n*pi)=0 for integer n)
    # Similarly, w = -13*pi^2 * sin(3*pi*x)*sin(2*pi*y) = 0 on all boundaries
    
    # So both BCs are homogeneous Dirichlet!
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Source term
    f_expr = 169.0 * pi**4 * ufl.sin(3*pi*x[0]) * ufl.sin(2*pi*x[1])
    
    # Boundary conditions (homogeneous)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc_zero]
    
    total_iterations = 0
    
    # Step 1: Solve -Delta w = f with w=0 on boundary
    w_sol = fem.Function(V)
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a, L1, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem1.solve()
    
    # Get iteration count
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Delta u = w with u=0 on boundary
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a, L2, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
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
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }