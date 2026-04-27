import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import os

os.environ["OMP_NUM_THREADS"] = "1"


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    nu_val = case_spec["pde"]["viscosity"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution - balance accuracy and time
    N = 160
    degree_u = 2
    degree_p = 1
    
    # Create mesh on unit square
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Parse boundary conditions from case_spec
    bc_data = case_spec["pde"]["boundary_conditions"]
    
    # Map boundary labels to geometric markers
    boundary_markers = {
        "y0": lambda x: np.isclose(x[1], ymin),
        "y1": lambda x: np.isclose(x[1], ymax),
        "x0": lambda x: np.isclose(x[0], xmin),
        "x1": lambda x: np.isclose(x[0], xmax),
    }
    
    bcs = []
    
    # Extract BC value from various possible formats
    def get_bc_value(bc_info):
        if isinstance(bc_info, dict):
            val = bc_info.get("value", [0.0, 0.0])
        elif isinstance(bc_info, (list, tuple)):
            val = bc_info
        else:
            val = [0.0, 0.0]
        return [float(v) for v in val]
    
    # Apply BCs: no-slip walls first, then moving lids
    # Order: y0 (bottom, no-slip), x0 (left, no-slip), x1 (right, moving), y1 (top, moving)
    bc_order = ["y0", "x0", "x1", "y1"]
    
    for label in bc_order:
        if label in bc_data and label in boundary_markers:
            val = get_bc_value(bc_data[label])
            facets = mesh.locate_entities_boundary(msh, fdim, boundary_markers[label])
            u_func = fem.Function(V)
            v0, v1 = val[0], val[1]
            u_func.interpolate(lambda x, v0=v0, v1=v1: np.vstack([
                np.full(x.shape[1], v0),
                np.full(x.shape[1], v1)
            ]))
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bcs.append(fem.dirichletbc(u_func, dofs, W.sub(0)))
    
    # Pressure pin at corner (xmin, ymin) = 0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    bcs.append(bc_p)
    
    # Define the nonlinear problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Parse source term
    f_src = case_spec["pde"]["source_term"]
    f = fem.Constant(msh, PETSc.ScalarType((float(f_src[0]), float(f_src[1]))))
    
    # Residual form for steady NS
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    J_form = ufl.derivative(F_form, w)
    
    # Initialize with zero and apply BCs
    w.x.array[:] = 0.0
    petsc.set_bc(w.x.petsc_vec, bcs)
    w.x.scatter_forward()
    
    # Solve nonlinear problem with Newton + direct LU
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_vals_flat = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, orig_idx in enumerate(eval_map):
            u_vals_flat[orig_idx] = vals[idx]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_vals_flat[:, 0]**2 + u_vals_flat[:, 1]**2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [4],
    }
    
    return {
        "u": vel_mag_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.18,
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": {
                "y1": {"type": "dirichlet", "value": [1.0, 0.0]},
                "x1": {"type": "dirichlet", "value": [0.0, -0.6]},
                "x0": {"type": "dirichlet", "value": [0.0, 0.0]},
                "y0": {"type": "dirichlet", "value": [0.0, 0.0]},
            },
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Shape: {u_grid.shape}")
    print(f"Min vel mag: {np.min(u_grid):.6f}")
    print(f"Max vel mag: {np.max(u_grid):.6f}")
    print(f"Mean vel mag: {np.mean(u_grid):.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
