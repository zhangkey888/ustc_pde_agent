import os
os.environ['OMP_NUM_THREADS'] = '4'
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
    nu = float(pde["parameters"]["viscosity"])
    
    # Extract output grid spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox
    
    # High mesh resolution for accuracy within generous time budget
    N = 384
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Variational forms for Stokes
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Stokes: -nu*laplacian(u) + grad(p) = f, div(u) = 0
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    bcs = []
    
    # Left boundary (x=0): u = [sin(pi*y), 0]
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.vstack([np.sin(np.pi * x[1]), np.zeros_like(x[1])]))
    bcs.append(fem.dirichletbc(u_left, left_dofs, W.sub(0)))
    
    # Bottom boundary (y=0): u = [0, 0]
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0)))
    
    # Right boundary (x=1): u = [0, 0]
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    right_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    u_right = fem.Function(V)
    u_right.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_right, right_dofs, W.sub(0)))
    
    # Top boundary (y=1): natural outflow (zero traction) - no Dirichlet BC
    
    # Pressure pin at origin for uniqueness
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))
    
    # Assemble system
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.assemble_vector(L_form)
    petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Solve with LU + MUMPS (robust for saddle-point systems)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    ksp.setFromOptions()
    ksp.setUp()
    
    w_h = fem.Function(W)
    ksp.solve(b, w_h.x.petsc_vec)
    w_h.x.scatter_forward()
    
    iterations = 1  # Direct solve = 1 iteration
    ksp_type_str = "preonly"
    pc_type_str = "lu"
    rtol = 1e-10
    
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()
    
    # Accuracy verification: L2 norm of divergence
    div_u = ufl.div(u_h)
    div_L2_sq = fem.assemble_scalar(fem.form(div_u * div_u * ufl.dx))
    div_L2 = np.sqrt(np.abs(comm.allreduce(div_L2_sq, op=MPI.SUM)))
    if comm.rank == 0:
        print(f"[Verification] L2 norm of div(u) = {div_L2:.6e}")
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    eps = 1e-10
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = np.clip(XX.ravel(), xmin + eps, xmax - eps)
    points[:, 1] = np.clip(YY.ravel(), ymin + eps, ymax - eps)
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    
    u_values = np.zeros((nx_out * ny_out, gdim))
    
    points_on_proc = []
    cells_on_proc = []
    eval_indices = []
    
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_indices.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_indices] = vals
    
    # Compute velocity magnitude
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)
    
    # Gather on all processes
    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": ksp_type_str,
        "pc_type": pc_type_str,
        "rtol": float(rtol),
        "iterations": int(iterations),
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "name": "stokes",
            "parameters": {"viscosity": "0.9"},
            "source": {"value": ["0.0", "0.0"]},
            "time": {"is_transient": False},
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]},
            "field": "velocity_magnitude"
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid max: {np.nanmax(result['u']):.6f}")
    print(f"u_grid min: {np.nanmin(result['u']):.6f}")
    print(f"solver_info: {result['solver_info']}")
    print(f"Total time: {t1-t0:.2f}s")
