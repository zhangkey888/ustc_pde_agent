import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Parameters
    epsilon = 0.01
    beta_val = [15.0, 0.0]

    mesh_resolution = 128
    element_degree = 2

    # Mesh creation
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Source term f
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f = -epsilon * ufl.div(ufl.grad(u_exact)) + beta_val[0] * ufl.grad(u_exact)[0] + beta_val[1] * ufl.grad(u_exact)[1]

    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))

    # Weak form
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    # SUPG Stabilization
    # cell diameter
    h = ufl.CellDiameter(domain)
    # Norm of beta
    vnorm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG parameter tau
    # tau = h / (2 * vnorm)
    tau = h / (2.0 * vnorm + 1e-12)
    
    # Residual
    R = -eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    # Test function convection derivative
    v_conv = ufl.dot(beta, ufl.grad(v))
    
    a_stab = a + tau * ufl.inner(ufl.dot(beta, ufl.grad(u)) - eps_const * ufl.div(ufl.grad(u)), v_conv) * ufl.dx
    L_stab = L + tau * ufl.inner(f, v_conv) * ufl.dx

    # Setup linear problem
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    problem = petsc.LinearProblem(
        a_stab, L_stab, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    
    # Get iterations
    solver_obj = problem.solver
    iterations = solver_obj.getIterationNumber()

    # Interpolate onto grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points_array = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(points_array.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    # If running in parallel, sum up the values
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape((ny, nx))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }

    return {"u": u_grid, "solver_info": solver_info}
