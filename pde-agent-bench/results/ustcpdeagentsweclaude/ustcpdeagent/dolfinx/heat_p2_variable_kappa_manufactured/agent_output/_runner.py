import argparse
import json
import time
import importlib.util
import numpy as np

def _load_module(path):
    spec = importlib.util.spec_from_file_location("agent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _get_solver_fn(module):
    if hasattr(module, "solve") and callable(module.solve):
        return module.solve
    if hasattr(module, "solve_case") and callable(module.solve_case):
        return module.solve_case
    raise AttributeError("Expected solve(case_spec) or solve_case(case_spec) in agent script")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.case) as f:
        case_spec = json.load(f)

    solver = _get_solver_fn(_load_module(args.script))

    t0 = time.time()
    result = solver(case_spec)
    t1 = time.time()

    if not isinstance(result, dict):
        raise ValueError("solve() must return a dict with keys: u (or u_grid) and solver_info")

    u_grid = result.get("u")
    if u_grid is None:
        u_grid = result.get("u_grid")
    solver_info = result.get("solver_info", {})

    if u_grid is None:
        raise ValueError("solve() returned no solution array")
    if not isinstance(solver_info, dict) or not solver_info:
        raise ValueError("solve() must return non-empty solver_info dict")

    required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol"]
    missing = [k for k in required_keys if k not in solver_info]
    if missing:
        raise ValueError(f"solver_info missing required keys: {missing}")

    u_grid = np.array(u_grid, dtype=float)

    # case_spec 是 agent 视图，output 直接在顶层
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    if u_grid.ndim == 1:
        # 1-D 时尝试按 (ny, nx) reshape
        if u_grid.size == ny * nx:
            u_grid = u_grid.reshape((ny, nx))
        else:
            side = int(round(u_grid.size ** 0.5))
            if side * side == u_grid.size:
                u_grid = u_grid.reshape((side, side))
            else:
                raise ValueError(f"Cannot reshape 1-D array of size {u_grid.size} into a 2-D grid")
    if u_grid.ndim != 2:
        raise ValueError(f"u_grid must be 2-D, got shape {u_grid.shape}")

    # 约定：u_grid 形状为 (ny, nx)，即 u_grid[row_y, col_x]
    # 若 agent 返回的尺寸与 oracle 不同，则重采样到 (ny, nx)
    # NaN（域外点）用 fill_value=np.nan 保留，不向内部晕染
    if u_grid.shape != (ny, nx):
        from scipy.interpolate import RegularGridInterpolator
        agent_ny, agent_nx = u_grid.shape          # 行=y，列=x
        agent_y = np.linspace(bbox[2], bbox[3], agent_ny)
        agent_x = np.linspace(bbox[0], bbox[1], agent_nx)
        interp = RegularGridInterpolator(
            (agent_y, agent_x), u_grid,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        ref_y = np.linspace(bbox[2], bbox[3], ny)
        ref_x = np.linspace(bbox[0], bbox[1], nx)
        YY, XX = np.meshgrid(ref_y, ref_x, indexing="ij")  # shape (ny, nx)
        u_grid = interp(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(ny, nx)

    x = np.linspace(bbox[0], bbox[1], nx)
    y = np.linspace(bbox[2], bbox[3], ny)
    np.savez(f"{args.outdir}/solution.npz", x=x, y=y, u=u_grid)

    # Save u.npy for specialized metrics (e.g., front propagation speed)
    np.save(f"{args.outdir}/u.npy", u_grid)

    # Save u_initial.npy if provided (for time-dependent problems)
    u_initial = result.get("u_initial")
    if u_initial is not None:
        u_initial = np.array(u_initial, dtype=float)
        if u_initial.ndim == 1:
            if u_initial.size == ny * nx:
                u_initial = u_initial.reshape((ny, nx))
            else:
                side = int(round(u_initial.size ** 0.5))
                if side * side == u_initial.size:
                    u_initial = u_initial.reshape((side, side))
        if u_initial.ndim == 2 and u_initial.shape != (ny, nx):
            from scipy.interpolate import RegularGridInterpolator
            init_ny, init_nx = u_initial.shape
            init_y = np.linspace(bbox[2], bbox[3], init_ny)
            init_x = np.linspace(bbox[0], bbox[1], init_nx)
            interp_init = RegularGridInterpolator(
                (init_y, init_x), u_initial,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            ref_y = np.linspace(bbox[2], bbox[3], ny)
            ref_x = np.linspace(bbox[0], bbox[1], nx)
            YY, XX = np.meshgrid(ref_y, ref_x, indexing="ij")
            u_initial = interp_init(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(ny, nx)
        np.save(f"{args.outdir}/u_initial.npy", u_initial)

    meta = {
        "wall_time_sec": t1 - t0,
        "solver_info": solver_info,
    }
    with open(f"{args.outdir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
