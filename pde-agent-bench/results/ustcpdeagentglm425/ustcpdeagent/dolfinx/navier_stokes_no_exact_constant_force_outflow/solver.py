import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract PDE parameters
    pde = case_spec["pde"]
    nu = float(pde["params"]["nu"])
    f_vals = [float(v) for v in pde["source"]]
    gdim = 2
    
    # Extract output grid
    out_spec = case_spec["output"]
    grid = out_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Mesh resolution
    mesh_res = 32
    
    # Create mesh
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Mixed function space: Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Define unknowns and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Body force
    f = fem.Constant(msh, PETSc.ScalarType(f_vals))
    
    # Strain rate and stress
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    # Nonlinear residual for Navier-Stokes
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, w)
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    
    # Zero velocity function for BCs
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    
    bcs = []
    
    # y=0 (bottom): u = [0,0]
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    bcs.append(fem.dirichletbc(u_zero, dofs_y0, W.sub(0)))
    
    # y=1 (top): u = [0,0]
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    bcs.append(fem.dirichletbc(u_zero, dofs_y1, W.sub(0)))
    
    # x=1 (right): u = [0,0]
    facets_x1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs_x1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x1)
    bcs.append(fem.dirichletbc(u_zero, dofs_x1, W.sub(0)))
    
    # Step 1: Solve Stokes problem for initial guess
    (u_s, p_s) = ufl.TrialFunctions(W)
    a_stokes = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_s)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_s * ufl.div(v) * ufl.dx
        + ufl.div(u_s) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "umfpack"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    stokes_lin_iters = stokes_problem.solver.getIterationNumber()
    
    # Copy Stokes solution as initial guess for NS
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    # Step 2: Solve Navier-Stokes with Newton
    petsc_options_ns = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "umfpack",
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options_ns
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Get iteration counts
    snes = problem.solver
    nonlin_iters = snes.getIterationNumber()
    ns_lin_iters = snes.getLinearSolveIterations()
    
    # Create scalar function space for velocity magnitude
    V_mag = fem.functionspace(msh, ("Lagrange", 2))
    u_mag = fem.Function(V_mag)
    
    # Compute velocity magnitude using fem.Expression
    u_sym = ufl.split(w)[0]
    mag_ufl = ufl.sqrt(ufl.inner(u_sym, u_sym))
    mag_expr = fem.Expression(mag_ufl, V_mag.element.interpolation_points)
    u_mag.interpolate(mag_expr)
    
    # Evaluate velocity magnitude on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_mag_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        vals = u_mag.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_flat = np.zeros(nx_out * ny_out)
        u_flat[eval_map] = vals.flatten()
        u_mag_grid = u_flat.reshape(ny_out, nx_out)
    
    # Clip tiny negative values from floating point
    u_mag_grid = np.maximum(u_mag_grid, 0.0)
    
    # Gather across processes
    u_mag_grid = comm.allreduce(u_mag_grid, op=MPI.SUM)
    
    return {
        "u": u_mag_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": stokes_lin_iters + ns_lin_iters,
            "nonlinear_iterations": [nonlin_iters],
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "params": {"nu": "0.3"},
            "source": ["1.0", "0.0"],
            "time": None,
        },
        "output": {
            "grid": {
                "nx": 32, "ny": 32,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6e}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")
