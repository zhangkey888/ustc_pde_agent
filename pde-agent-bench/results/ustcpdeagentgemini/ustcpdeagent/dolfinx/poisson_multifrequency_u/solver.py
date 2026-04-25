import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    # 1. Setup mesh and function space
    comm = MPI.COMM_WORLD
    
    # Grid specification
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Choose mesh resolution based on time budget and accuracy constraints.
    # P2 elements with 64x64 resolution will be very accurate and extremely fast.
    mesh_resolution = 64
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 2. Boundary conditions
    # For this manufactured solution, u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                             np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
                             
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 3. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x_ufl = ufl.SpatialCoordinate(domain)
    # Exact solution: u = sin(pi*x)*sin(pi*y) + 0.3*sin(6*pi*x)*sin(6*pi*y)
    # f = -div(grad(u))
    f = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi*x_ufl[0]) * ufl.sin(ufl.pi*x_ufl[1]) + \
        21.6 * ufl.pi**2 * ufl.sin(6.0*ufl.pi*x_ufl[0]) * ufl.sin(6.0*ufl.pi*x_ufl[1])
        
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 4. Solve the linear system
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    
    # Get the iterations (since petsc.LinearProblem doesn't expose it directly easily, 
    # we can just estimate or use manual KSP if strict tracking is needed, but here we provide a dummy/typical value 
    # as we used LinearProblem which is fine for basic metrics).
    # To be precise, let's create a manual KSP solver to track iterations accurately.
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
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    iterations = solver.getIterationNumber()
    
    # 5. Output extraction
    # Create the regular grid
    x_line = np.linspace(bbox[0], bbox[1], nx_out)
    y_line = np.linspace(bbox[2], bbox[3], ny_out)
    X, Y = np.meshgrid(x_line, y_line)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Probe points
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
            
    u_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
