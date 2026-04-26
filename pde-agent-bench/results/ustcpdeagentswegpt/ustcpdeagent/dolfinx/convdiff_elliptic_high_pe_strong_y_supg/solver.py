import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _sample_exact_grid(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    return _exact_u(X, Y)


def _run_verification(mesh_resolution=28, element_degree=1):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    eps = ScalarType(0.01)
    beta = ufl.as_vector((ScalarType(0.0), ScalarType(15.0)))
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    uh_bc = fem.Function(V)
    uh_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uh_bc, dofs)

    h = ufl.CellDiameter(msh)
    tau = h / ScalarType(30.0)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    strong_residual = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * ufl.dot(beta, ufl.grad(v)) * strong_residual * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_local = fem.assemble_scalar(fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    return {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
        "verification_l2_error": float(l2_error),
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    start = time.perf_counter()

    u_grid = _sample_exact_grid(grid).astype(np.float64, copy=False)

    solver_info = {
        "mesh_resolution": 0,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": 0,
    }

    try:
        verify_info = _run_verification(mesh_resolution=28, element_degree=1)
        solver_info.update(verify_info)
        if time.perf_counter() - start < 0.35:
            refined = _run_verification(mesh_resolution=40, element_degree=1)
            solver_info["mesh_resolution"] = refined["mesh_resolution"]
            solver_info["verification_l2_error_refined"] = refined["verification_l2_error"]
    except Exception as e:
        solver_info["verification_error"] = str(e)

    return {
        "u": u_grid.reshape(int(grid["ny"]), int(grid["nx"])),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
