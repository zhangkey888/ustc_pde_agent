import numpy as np
from mpi4py import MPI as mpi_MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = mpi_MPI.COMM_WORLD

    # Extract output grid spec
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]  # [xmin, xmax, ymin, ymax]

    # Extract PDE parameters
    pde = case_spec["pde"]
    kappa_val = pde.get("coefficients", {}).get("kappa", 1.0)
    f_val = pde.get("source", 1.0)
    
    # Solver parameters - balanced for accuracy and speed
    mesh_resolution = 256
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    # Create mesh on [0,1]x[0,1] with quadrilaterals
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    f = fem.Constant(domain, ScalarType(f_val))

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "pc_hypre_type": "boomeramg",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros((pts.shape[0],))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    # Gather across processes
    if comm.size > 1:
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=mpi_MPI.SUM)
        u_grid = u_global.reshape(ny_out, nx_out)
    else:
        u_grid = u_values.reshape(ny_out, nx_out)

    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    # Check if time-dependent and add time info if needed
    time_info = pde.get("time", None)
    if time_info:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
