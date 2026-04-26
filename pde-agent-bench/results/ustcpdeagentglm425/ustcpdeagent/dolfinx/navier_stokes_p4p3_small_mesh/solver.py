import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    # Parse case spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Get viscosity
    nu = 0.2
    try:
        nu = float(case_spec["pde"]["parameters"]["viscosity"])
    except (KeyError, TypeError):
        pass
    
    # Mesh resolution
    n_mesh = 32
    
    # Create mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n_mesh, n_mesh, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Mixed element: P2/P1 Taylor-Hood
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    
    # Manufactured solution (UFL expressions)
    u_ex = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term: f = (u_ex·∇)u_ex - ν∇²u_ex + ∇p_ex
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    
    # Define nonlinear residual
    # Using: -ν∇²u + (u·∇)u + ∇p = f, ∇·u = 0
    # Weak form:
    # 2ν(ε(u),ε(v)) + ((u·∇)u, v) - (p, ∇·v) - (q, ∇·u) = (f, v)
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, w)
    
    # Boundary conditions: u = u_ex on ∂Ω
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_ex, V.element.interpolation_points)
    )
    bc_u = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))
    
    # Pressure pin at origin: p(0,0) = p_ex(0,0) = cos(0)*cos(0) = 1
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.interpolate(
        fem.Expression(p_ex, Q.element.interpolation_points)
    )
    # Only add pressure BC if there are dofs on this process
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Initial guess: interpolate exact solution for fast convergence
    w.sub(0).interpolate(
        fem.Expression(u_ex, V.element.interpolation_points)
    )
    w.sub(1).interpolate(
        fem.Expression(p_ex, Q.element.interpolation_points)
    )
    
    # Solve with Newton's method
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Get iteration info
    snes = problem.snes
    newton_iters = snes.getIterationNumber()
    ksp_total_iters = snes.getLinearSolveIterations()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Compute L2 error for verification
    L2_sq = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx))
    L2_sq = msh.comm.allreduce(L2_sq, op=MPI.SUM)
    L2_error = np.sqrt(L2_sq) if L2_sq > 0 else 0.0
    
    if comm.rank == 0:
        print(f"L2 velocity error: {L2_error:.6e}")
        print(f"Newton iterations: {newton_iters}")
        print(f"Total linear iterations: {ksp_total_iters}")
    
    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute magnitude
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": n_mesh,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": ksp_total_iters,
        "nonlinear_iterations": [newton_iters],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    # Quick test
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {
            "parameters": {"viscosity": 0.2},
            "time": {"is_transient": False},
        }
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.nanmax(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
