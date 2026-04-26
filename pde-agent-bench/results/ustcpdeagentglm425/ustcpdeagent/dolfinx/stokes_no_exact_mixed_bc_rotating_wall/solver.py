import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde = case_spec.get("pde", {})
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Get viscosity
    nu = 1.0
    if "parameters" in pde and "viscosity" in pde["parameters"]:
        nu = float(pde["parameters"]["viscosity"])
    elif "coefficients" in pde and "viscosity" in pde["coefficients"]:
        nu = float(pde["coefficients"]["viscosity"])
    
    # Source term
    f_vals = [0.0, 0.0]
    if "source" in pde:
        f_vals = [float(v) for v in pde["source"]]
    
    # Mesh resolution - use 256 for good accuracy within time budget
    mesh_res = 256
    gdim = 2
    
    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, np.array(f_vals, dtype=PETSc.ScalarType))
    
    # Stokes weak form
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    bcs = []
    
    # Left wall (x = xmin): u = [0.0, 0.0]
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], xmin))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_left, left_dofs, W.sub(0)))
    
    # Bottom wall (y = ymin): u = [0.0, 0.0]
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], ymin))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0)))
    
    # Top wall (y = ymax): u = [0.5, 0.0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], ymax))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([0.5 * np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))
    
    # Right wall (x = xmax): natural BC (traction-free outflow)
    # No Dirichlet BC needed
    
    # Solve using LinearProblem with direct MUMPS solver
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    
    # Get iterations
    ksp = problem.solver
    its = ksp.getIterationNumber()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    
    # Sample velocity magnitude on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Build bounding box tree for point evaluation
    bb_tree = geometry.bb_tree(msh, tdim)
    
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
    
    u_mag_values = np.zeros((nx_out * ny_out,), dtype=np.float64)
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_eval, cells_eval)  # shape (N, gdim)
        magnitude = np.linalg.norm(u_vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            u_mag_values[global_idx] = magnitude[idx]
    
    # Gather across processes if parallel
    u_mag_values = comm.allreduce(u_mag_values, op=MPI.SUM)
    
    u_grid = u_mag_values.reshape((ny_out, nx_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": its,
        }
    }

if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"type": "stokes", "parameters": {"viscosity": 1.0}, "source": ["0.0", "0.0"]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"u shape: {result['u'].shape}")
    print(f"u max: {result['u'].max():.6f}")
    print(f"u min: {result['u'].min():.6e}")
    print(f"Total time: {t1-t0:.2f}s")
    print(f"solver_info: {result['solver_info']}")
