#!/usr/bin/env python3
"""
PDEBench Parallel Multi-Worker Benchmark Runner

Runs benchmark cases concurrently across multiple workers, each of which may
use a different agent / model backend.  All workers share one thread pool so
that total parallelism is bounded.

Usage:
    python run_benchmark_parallel.py --config configs/parallel.yaml
    python run_benchmark_parallel.py --config configs/parallel.yaml --timeout 300
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.core.prompt_builder import generate_prompt
from pdebench.core.llm_client import call_llm, LLMClient
from pdebench.analysis import GateAnalyzer
from pdebench.metrics.specialized import get_specialized_metrics_computer
from pdebench.agents import AgentRegistry, get_agent

from run_benchmark import (
    load_agent_config,
    load_benchmark_cases,
    run_oracle,
    execute_solver,
    compute_error,
    compute_summary,
    print_summary,
    _make_error_result,
    _write_oracle_reference,
)

# ═══════════════════════════════════════════════════════════════════════════════
# defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: dict = {
    "data_file":      "data/benchmark_v2.jsonl",
    "output_dir":     "results/",
    "solver_library": "dolfinx",
    "version":        None,
    "timeout":        300,
    "cases":          [],
    "equation_types": [],
    "workers":        [],
}


# ═══════════════════════════════════════════════════════════════════════════════
# config loading  (YAML → JSON → simple parser)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_load_yaml(path: Path) -> dict:
    """Read *path* and return a dict.  Tries yaml, then json, then dies."""
    raw = path.read_text()
    try:
        import yaml
        return yaml.safe_load(raw) or {}
    except ImportError:
        pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    raise SystemExit(
        f"Cannot parse config: {path}.  "
        "Install pyyaml (`pip install pyyaml`) or use JSON format."
    )


def load_config(config_path: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(_try_load_yaml(config_path))
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# thread‑safe helpers
# ═══════════════════════════════════════════════════════════════════════════════

_print_lock = threading.Lock()
_progress_lock = threading.Lock()


def sprint(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


def _substitute_env_vars(data: dict) -> dict:
    """Replace ``${VAR_NAME}`` placeholders in every string value of *data*.

    Operates on a JSON-serialised round-trip so nested dicts are handled.
    """
    try:
        text = json.dumps(data)
    except (TypeError, ValueError):
        return data

    for match in re.finditer(r'\$\{([^}]+)\}', text):
        val = os.environ.get(match.group(1), "")
        text = text.replace(match.group(0), val)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return data


def _resolve_agent_config(worker: dict, agent_name: str) -> dict:
    """Return the agent config dict for *worker*.

    Priority:  worker.agent_config  >  pdebench/configs/{agent}.json  >  {}.
    """
    if worker.get("agent_config"):
        return _substitute_env_vars(worker["agent_config"])
    try:
        return load_agent_config(agent_name)
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# single‑task runner  (executed in a thread‑pool thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_task(
    case:             Dict,
    worker:           dict,
    root_output_dir:  Path,
    oracle_cache_dir: Path,
    solver_library:   str,
    use_docker:       bool,
    docker_image:     Optional[str],
    progress_state:   dict,
) -> Dict:
    """Full pipeline for one **(case, worker)** pair."""

    worker_name = worker["name"]
    agent_name  = worker["agent"]
    timeout     = worker["timeout"]

    skip_generation   = worker.get("skip_generation", False)
    eval_existing_dir = worker.get("eval_existing_dir")

    is_code_agent = AgentRegistry.is_registered(agent_name)
    is_llm        = agent_name in LLMClient.SUPPORTED_AGENTS

    case_id      = case["id"]
    case_output  = root_output_dir / worker_name / solver_library / case_id
    case_output.mkdir(parents=True, exist_ok=True)

    # ── 1. oracle (always cached after Phase 2) ──────────────────────────
    oracle_info = run_oracle(
        case, oracle_cache_dir,
        solver_library=solver_library,
        use_docker=use_docker, docker_image=docker_image,
    )

    oracle_output = case_output / "oracle_output"
    oracle_output.mkdir(parents=True, exist_ok=True)
    _write_oracle_reference(case, oracle_info, oracle_output)

    # ── 2. prompt ──────────────────────────────────────────────────────
    prompt = generate_prompt(case, oracle_info, solver_library=solver_library)
    (case_output / "prompt.md").write_text(prompt)

    # ── 3. obtain solver code ───────────────────────────────────────────
    response    = None
    solver_path = case_output / "solver.py"

    try:
        # --- pre‑existing solver from disk ---
        if eval_existing_dir:
            existing = Path(eval_existing_dir)
            if not existing.is_absolute():
                existing = (Path(__file__).parent.parent / existing).resolve()
            src = existing / case_id / "solver.py"
            if not src.exists():
                return _make_error_result(
                    case_id, "SOLVER_NOT_FOUND",
                    f"No solver at {src}",
                    case_output=case_output, case=case,
                )
            solver_code = src.read_text()

        elif skip_generation and solver_path.exists():
            solver_code = solver_path.read_text()

        # --- Code Agent (mini-swe-agent, openhands, …) ---
        elif is_code_agent:
            agent_cfg = _resolve_agent_config(worker, agent_name)
            agent = get_agent(agent_name, config=agent_cfg)
            response = agent.generate_solution(
                prompt=prompt,
                context={
                    "case_id":     case_id,
                    "case_spec":   case,
                    "oracle_info": oracle_info,
                },
            )
            agent.cleanup()
            if not response.success:
                return _make_error_result(
                    case_id, "AGENT_ERROR", response.error or "unknown",
                    case_output=case_output, case=case,
                )
            solver_code = response.code
            (case_output / "agent_response.txt").write_text(
                response.raw_response or ""
            )

        # --- plain LLM (gpt-4o, sonnet-3.5, …) ---
        elif is_llm:
            response = call_llm(agent_name, prompt)
            if not response.success:
                return _make_error_result(
                    case_id, "LLM_ERROR", response.error or "unknown",
                    case_output=case_output, case=case,
                )
            solver_code = response.code
            (case_output / "llm_response.txt").write_text(
                response.raw_response or ""
            )

        else:
            return _make_error_result(
                case_id, "INVALID_AGENT",
                f"Unknown agent: {agent_name}",
                case_output=case_output, case=case,
            )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return _make_error_result(
            case_id,
            "AGENT_ERROR" if is_code_agent else "LLM_ERROR",
            str(exc),
            case_output=case_output, case=case,
        )

    # ── 4. execute solver in sandbox ────────────────────────────────────
    exec_result = execute_solver(
        solver_code, case, case_output, timeout,
        solver_library=solver_library,
        use_docker=use_docker, docker_image=docker_image,
    )
    if not exec_result["success"]:
        return _make_error_result(
            case_id, "EXECUTION_ERROR",
            exec_result.get("error_message"),
            case_output=case_output, case=case,
        )

    # ── 5. compute error ────────────────────────────────────────────────
    error = compute_error(exec_result["agent_output"], oracle_info)
    if np.isnan(error):
        return _make_error_result(
            case_id, "EVALUATION_ERROR", "Error computed as NaN",
            case_output=case_output, case=case,
        )

    # ── 6. gate analysis (accuracy → time) ──────────────────────────────
    eval_cfg      = case.get("evaluation_config", {})
    legacy_tol    = eval_cfg.get("tolerance", 1.2)
    acc_tol       = eval_cfg.get("accuracy_tolerance", legacy_tol)
    time_tol      = eval_cfg.get("time_tolerance", legacy_tol)
    MIN_ERR       = 1e-6

    target_error = max(oracle_info["error"] * acc_tol, MIN_ERR)
    target_time  = oracle_info["time"] * time_tol

    if error > target_error:
        status = "FAIL"
        fail_reason = f"ACCURACY_FAIL: error={error:.2e} > target={target_error:.2e}"
    elif exec_result["time"] > target_time:
        status = "FAIL"
        fail_reason = (
            f"TIME_FAIL: time={exec_result['time']:.3f}s "
            f"> target={target_time:.3f}s"
        )
    else:
        status = "PASS"
        fail_reason = None

    # ── 7. build result ─────────────────────────────────────────────────
    result: dict = {
        "case_id":          case_id,
        "worker":           worker_name,
        "equation_type":    case.get("pde_classification", {}).get("equation_type", "unknown"),
        "status":           status,
        "error":            error,
        "time":             exec_result["time"],
        "oracle_error":     oracle_info["error"],
        "oracle_time":      oracle_info["time"],
        "tolerance":        legacy_tol,
        "accuracy_tolerance": acc_tol,
        "time_tolerance":   time_tol,
        "target_error":     target_error,
        "target_time":      target_time,
        "fail_reason":      fail_reason,
    }

    # gate breakdown
    ga = GateAnalyzer()
    gb = ga.analyze_single_case(
        case_id=case_id,
        exec_result={"success": True, "error": error, "time": exec_result["time"]},
        eval_result={
            "target_error": target_error,
            "target_time":  target_time,
            "fail_reason":  fail_reason,
            "status":       status,
        },
        oracle_info=oracle_info,
    )
    result["gate_breakdown"] = {
        "exec_valid":    gb.exec_valid,
        "accuracy_pass": gb.accuracy_pass,
        "time_pass":     gb.time_pass,
        "final_pass":    gb.final_pass,
        "failure_stage": gb.failure_stage,
        "failure_reason": gb.failure_reason,
    }

    # llm usage (if available)
    if response is not None and hasattr(response, "usage") and response.usage:
        result["llm_usage"] = response.usage

    # specialised math-type metrics
    math_types = case.get("pde_classification", {}).get("math_type", [])
    mt_metrics: dict = {}
    for mt in math_types:
        computer = get_specialized_metrics_computer(
            mt, exec_result["agent_output"], oracle_output, case,
        )
        if computer is None:
            continue
        mt_metrics[mt] = computer.compute({
            "runtime_sec":  exec_result["time"],
            "error":        error,
            "test_params":  {},
        })
    if mt_metrics:
        result["math_types"]        = math_types
        result["math_type_metrics"] = mt_metrics

    # write per‑case result
    with open(case_output / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── progress line ───────────────────────────────────────────────────
    with _progress_lock:
        progress_state["completed"] += 1
        done  = progress_state["completed"]
        total = progress_state["total"]

    sprint(
        f"[{worker_name:20s}] [{done:4d}/{total:4d}] "
        f"{case_id:36s} {status:4s}  "
        f"err={error:.2e}  t={exec_result['time']:.2f}s"
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_parallel(config: dict) -> None:
    root_dir = Path(__file__).parent.parent

    # ── resolve paths ───────────────────────────────────────────────────
    data_file  = Path(config["data_file"])
    output_dir = Path(config["output_dir"])
    if not data_file.is_absolute():
        data_file = root_dir / data_file
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    solver_library = config["solver_library"]

    # dataset version shorthand
    if config.get("version") == "v1":
        data_file = root_dir / "data" / "benchmark_v1.jsonl"
        if solver_library != "dolfinx":
            print("Error: v1 dataset only supports dolfinx backend.")
            sys.exit(1)
    elif config.get("version") == "v2":
        data_file = root_dir / "data" / "benchmark_v2.jsonl"

    if not data_file.exists():
        print(f"Error: data file not found: {data_file}")
        sys.exit(1)

    global_cases   = config.get("cases") or None
    global_eq      = config.get("equation_types") or None
    global_timeout = config.get("timeout", 300)

    use_docker  = solver_library in ("dealii", "firedrake")
    docker_image = None

    # ── validate & normalise workers ────────────────────────────────────
    raw_workers: list = config.get("workers", [])
    if not raw_workers:
        print("Error: no workers defined in config.")
        sys.exit(1)

    seen_names: set = set()
    workers: List[dict] = []

    for w in raw_workers:
        name = w.get("name", "")
        if not name:
            print("Error: every worker must have a 'name' field.")
            sys.exit(1)
        if name in seen_names:
            print(f"Error: duplicate worker name '{name}'.")
            sys.exit(1)
        seen_names.add(name)

        agent = w.get("agent", "")
        code  = AgentRegistry.is_registered(agent)
        llm   = agent in LLMClient.SUPPORTED_AGENTS
        if not llm and not code:
            print(f"Error: unknown agent '{agent}' in worker '{name}'.")
            print(f"  LLMs:        {list(LLMClient.SUPPORTED_AGENTS.keys())}")
            print(f"  Code Agents: {AgentRegistry.list_agents()}")
            sys.exit(1)

        workers.append({
            "name":              name,
            "agent":             agent,
            "concurrency":       int(w.get("concurrency", 1)),
            "timeout":           int(w.get("timeout") or global_timeout),
            "cases":             w.get("cases") or None,
            "equation_types":    w.get("equation_types") or None,
            "skip_generation":   bool(w.get("skip_generation", False)),
            "eval_existing_dir": w.get("eval_existing_dir"),
            "max_attempts":      int(w.get("max_attempts", 1)),
            "agent_config":      w.get("agent_config"),
        })

    # ── header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PDEBench Parallel — Multi-Worker Evaluation")
    print("=" * 80)
    print(f"  Data:         {data_file}")
    print(f"  Output:       {output_dir}")
    print(f"  Library:      {solver_library}")
    print(f"  Workers:      {len(workers)}")
    for w in workers:
        tag = "Code Agent" if AgentRegistry.is_registered(w["agent"]) else "LLM"
        print(f"    · {w['name']:20s}  agent={w['agent']:18s}  "
              f"type={tag:10s}  concurrency={w['concurrency']}")
    print("=" * 80)

    # ── Phase 1: load cases ─────────────────────────────────────────────
    cases = load_benchmark_cases(
        data_file, global_cases, global_eq, solver_library=solver_library,
    )
    print(f"\n  Loaded {len(cases)} cases")
    if not cases:
        print("  No cases!")
        sys.exit(1)

    # ── Phase 2: pre‑compute oracles (serial, builds shared cache) ──────
    oracle_cache_dir = output_dir / ".oracle_cache" / solver_library
    oracle_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Phase 2 — Pre‑computing oracles ({len(cases)} cases)")
    print(f"{'─' * 60}")

    for i, case in enumerate(cases, 1):
        cid        = case["id"]
        cache_file = oracle_cache_dir / f"{cid}.json"
        if cache_file.exists():
            print(f"  [{i:3d}/{len(cases)}] {cid:40s}  (cached)")
        else:
            print(f"  [{i:3d}/{len(cases)}] {cid:40s}  computing...",
                  end="", flush=True)
            run_oracle(
                case, oracle_cache_dir,
                solver_library=solver_library,
                use_docker=use_docker, docker_image=docker_image,
            )
            print(" done")

    # ── Phase 3: build task queue ───────────────────────────────────────
    tasks: list = []          # [(case, worker), …]
    for w in workers:
        w_filtered = load_benchmark_cases(
            data_file,
            w.get("cases") or None,
            w.get("equation_types") or None,
            solver_library=solver_library,
        )
        for case in w_filtered:
            tasks.append((case, w))

    total_tasks      = len(tasks)
    total_concurrency = sum(w["concurrency"] for w in workers)

    print(f"\n{'─' * 60}")
    print(f"  Phase 3 — {total_tasks} tasks → pool of {total_concurrency} threads")
    print(f"{'─' * 60}")
    for w in workers:
        n = sum(1 for _, t_w in tasks if t_w is w)
        print(f"    · {w['name']:20s}  {n:4d} tasks  "
              f"×{w['concurrency']} concurrency")
    print(f"{'─' * 60}\n")

    # ── Phase 4: parallel execution ─────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_state = {"completed": 0, "total": total_tasks}
    all_results: Dict[str, List[Dict]] = {w["name"]: [] for w in workers}

    with ThreadPoolExecutor(max_workers=total_concurrency) as executor:
        futures = {}
        for case, worker in tasks:
            fut = executor.submit(
                _run_task,
                case, worker,
                output_dir,
                oracle_cache_dir,
                solver_library,
                use_docker,
                docker_image,
                progress_state,
            )
            futures[fut] = (case["id"], worker["name"])

        for future in as_completed(futures):
            cid, wname = futures[future]
            try:
                result = future.result()
                all_results.setdefault(wname, []).append(result)
            except Exception as exc:
                sprint(f"  [!] thread crash — {wname}/{cid}: {exc}")
                all_results.setdefault(wname, []).append({
                    "case_id":       cid,
                    "worker":        wname,
                    "status":        "THREAD_ERROR",
                    "error_message": str(exc),
                })

    # ── Phase 5: summaries ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Phase 5 — Summaries")
    print(f"{'─' * 60}")

    combined: dict = {}
    for wname, results in all_results.items():
        summary = compute_summary(wname, results)
        combined[wname] = summary

        wdir = output_dir / wname / solver_library
        wdir.mkdir(parents=True, exist_ok=True)
        with open(wdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print_summary(summary)

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 80)
    print("  Benchmark Complete!")
    print(f"  Results: {output_dir}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDEBench Parallel Multi-Worker Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", type=Path, required=True,
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Override global timeout",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Override global case filter",
    )
    parser.add_argument(
        "--equation-types", nargs="+", default=None,
        help="Override global equation-type filter",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides (global only)
    if args.timeout is not None:
        cfg["timeout"] = args.timeout
    if args.cases is not None:
        cfg["cases"] = args.cases
    if args.equation_types is not None:
        cfg["equation_types"] = args.equation_types

    # normalise Path → str
    for key in ("data_file", "output_dir"):
        if isinstance(cfg.get(key), Path):
            cfg[key] = str(cfg[key])

    run_benchmark_parallel(cfg)


if __name__ == "__main__":
    main()
