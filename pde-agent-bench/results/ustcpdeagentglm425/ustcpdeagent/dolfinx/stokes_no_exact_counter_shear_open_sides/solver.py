import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    nu_val = float(pde.get("nu", 1.0))
    f_vals = pde.get("f", ["0.0", "0.0"])
    out_spec = case_spec["output"]
    nx_out = out_spec["grid"]["nx"]
    ny_out = out_spec["grid"]["ny"]
    bbox = out_spec["grid"]["bbox"]
    mesh_res = 128

    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(msh, PETSc.ScalarType(np.array([float(f_vals[0]), float(f_vals[1])])))

    # Laplacian formulation: -nu*laplacian(u) + grad(p) = f
    a = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))

    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack([-np.ones(x.shape[1]), np.zeros(x.shape[1])]))

    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)

    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0))
    bcs = [bc_top, bc_bottom]

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

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
    w_h.x.scatter_forward()
    u_h = w_h.sub(0).collapse()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_array, cells_array)
        u_values[eval_map] = vals

    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    # Accuracy verification against analytical Couette flow: u_x = 2y-1, u_y = 0
    y_anal = YY.ravel()
    ux_anal = 2.0 * y_anal - 1.0
    uy_anal = np.zeros_like(y_anal)
    mag_anal = np.sqrt(ux_anal**2 + uy_anal**2).reshape(ny_out, nx_out)
    l2_err = np.sqrt(np.mean((magnitude - mag_anal)**2))
    max_err = np.max(np.abs(magnitude - mag_anal))
    if comm.rank == 0:
        print(f"L2 error vs Couette: {l2_err:.2e}, Max error: {max_err:.2e}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }
    pde_time = pde.get("time", None)
    if pde_time and pde_time.get("is_transient", False):
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    return {"u": magnitude, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 1.0, "f": ["0.0", "0.0"]},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    print(f"Output shape: {result[chr(117)].shape}")
    print(f"Solver info: {result[chr(115)+chr(111)+chr(108)+chr(118)+chr(101)+chr(114)+chr(95)+chr(105)+chr(110)+chr(102)+chr(111)]}")
