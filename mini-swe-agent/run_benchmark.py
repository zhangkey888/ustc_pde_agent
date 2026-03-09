import os
import shutil
import yaml
import typer
from pathlib import Path
from rich.console import Console

# 引入 MiniSWE 组件
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent import package_dir

app = typer.Typer()
console = Console()

# --- 这里填入你 pdeagentbench 的绝对路径 ---
# 例如: "/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/cases"
PDEBENCH_ROOT = "/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench"  # <--- 请修改这里 !!!

MASTER_INSTRUCTION = """
You are an expert computational scientist. Your task is to generate a complete Python solver script using FEniCSx (dolfinx) for a given PDE problem.

**Output Requirements:**
1. The script must be named `solver.py`.
2. It must accept the following command-line arguments:
   - `--resolution`: Integer, grid resolution (e.g., 128).
   - `--degree`: Integer, polynomial degree of the element.
   - `--outdir`: String, path to the output directory.
   - `--dt`: Float, time step size (only for time-dependent problems).

**Output Files:**
The script must execute and save the following two files in the folder specified by `--outdir`:
1. `solution.npz` (Required): Must contain `x`, `y`, and `u` arrays.
2. `meta.json` (Required): Must contain solver statistics.

**Library:** Use `dolfinx`, `ufl`, `mpi4py`, `petsc4py`.

IMPORTANT: Use the syntax compatible with FEniCSx (dolfinx) v0.8.0+. Do NOT use the legacy `dolfin` library or `Expression` class syntax.
DO NOT ask for clarification. Write the code, save it to `solver.py`, and finish.
"""

def get_full_prompt(case_description: str) -> str:
    return f"""
{MASTER_INSTRUCTION}

---
Here is the specific PDE problem you need to solve:

{case_description}

Please write the 'solver.py' file now.
"""

@app.command()
def main(
    pdebench_path: str = typer.Option(PDEBENCH_ROOT, help="Absolute path to pde-agent-bench root"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    target_case: str = typer.Option(None, "--case", "-c", help="Specific case name to run (e.g. poisson_simple)"),
):
    """
    Run MiniSWE Agent on PDEBench cases across directories.
    """
    
    # 1. 路径处理
    bench_root = Path(pdebench_path)
    if not bench_root.exists():
        console.print(f"[bold red]Error: Path not found: {bench_root}[/bold red]")
        return

    cases_dir = bench_root / "cases"
    results_dir = bench_root / "results" / "miniswe"

    if not cases_dir.exists():
        console.print(f"[bold red]Error: 'cases' directory not found in {bench_root}.[/bold red]")
        console.print("Did you run 'python scripts/build_cases.py ...' inside pde-agent-bench?")
        return

    # 2. 获取题目列表
    all_cases = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    
    # 过滤题目
    if target_case:
        all_cases = [p for p in all_cases if p.name == target_case]
        if not all_cases:
            console.print(f"[red]Case '{target_case}' not found![/red]")
            return

    console.print(f"[bold green]Found {len(all_cases)} cases to process.[/bold green]")

    # 3. Agent 配置
    default_config_path = package_dir / "config" / "default.yaml"
    base_config = yaml.safe_load(default_config_path.read_text())["agent"] if default_config_path.exists() else {}
    
    agent_config = {
        **base_config,
        "mode": "yolo", 
        "confirm_exit": False,
        "quiet": True 
    }

    original_cwd = os.getcwd()

    # 4. 循环运行
    for case_folder in all_cases:
        case_id = case_folder.name
        console.rule(f"[bold blue]Processing: {case_id}[/bold blue]")

        # 准备结果目录
        case_output_dir = results_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # 读取题目
        desc_file = case_folder / "description.md"
        if not desc_file.exists():
            console.print(f"[yellow]Skipping {case_id}: description.md missing[/yellow]")
            continue
        
        task_prompt = get_full_prompt(desc_file.read_text())

        # 切换工作目录到 results/miniswe/{case_id}
        os.chdir(case_output_dir)
        console.print(f"Directory switched to: {case_output_dir}")

        try:
            model_obj = LitellmModel(model_name=model)
            env = LocalEnvironment() # 使用当前目录
            agent = InteractiveAgent(model=model_obj, env=env, **agent_config)

            exit_status, result = agent.run(task_prompt)
            console.print(f"[green]Finished {case_id} ({exit_status})[/green]")

        except Exception as e:
            console.print(f"[bold red]Failed {case_id}: {e}[/bold red]")
            import traceback
            traceback.print_exc()
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    app()