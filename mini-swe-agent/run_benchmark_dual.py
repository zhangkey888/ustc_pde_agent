import os
import yaml
import typer
import shutil
from pathlib import Path
from rich.console import Console

# --- MiniSWE 组件 ---
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent import package_dir

# --- 自定义编排器 ---
# 确保 pde_orchestrator.py 在同一目录下，或者在 PYTHONPATH 中
try:
    from minisweagent.agents.pde_orchestrator import PDEOrchestrator
except ImportError:
    print("Error: Could not import PDEOrchestrator. Please ensure 'pde_orchestrator.py' is in the same directory.")
    exit(1)

app = typer.Typer()
console = Console()

# --- 全局常量 ---
PDEBENCH_ROOT = "/data5/store1/zky/pde-agent-bench"  # 你的数据根目录

# 这是一个总的格式约束，会拼接到每个具体 Case 的描述前
# 保持了对 solver.py, solution.npz, meta.json 的强制要求
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
    """将通用格式要求与具体题目描述拼接"""
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
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name (e.g., gpt-4o, claude-3-5-sonnet)"),
    target_case: str = typer.Option(None, "--case", "-c", help="Specific case name to run (e.g., diffusion_1d)"),
    task_prompts: str = typer.Option("config/pde_task_prompts.yaml", help="Path to the PDE-specific task prompts yaml"),
):
    """
    Run MiniSWE Dual-Agent System (Generator -> Verifier) on PDEBench cases.
    """
    
    # 1. 路径与环境检查
    bench_root = Path(pdebench_path)
    if not bench_root.exists():
        console.print(f"[bold red]Error: PDEBench root not found: {bench_root}[/bold red]")
        return

    cases_dir = bench_root / "cases"
    results_dir = bench_root / "results" / "miniswe_dual"  # 结果存放目录

    if not cases_dir.exists():
        console.print(f"[bold red]Error: 'cases' directory not found in {bench_root}.[/bold red]")
        return

    # 2. 检查 Task Prompts 文件是否存在
    task_prompts_path = Path(task_prompts)
    if not task_prompts_path.exists():
        # [新增逻辑] 尝试在 src/minisweagent/config/ 下查找
        # package_dir 指向 .../src/minisweagent
        # 我们取传入参数的文件名 (例如 pde_task_prompts.yaml) 并拼接到 config 目录
        pkg_config_path = package_dir / "config" / task_prompts_path.name
        
        if pkg_config_path.exists():
            task_prompts_path = pkg_config_path
            console.print(f"[dim]Loaded config from package: {task_prompts_path}[/dim]")
        else:
            # 如果包里也没有，尝试相对于当前工作目录查找 (fallback)
            cwd_path = Path(os.getcwd()) / task_prompts
            if cwd_path.exists():
                task_prompts_path = cwd_path
    # if not task_prompts_path.exists():
    #     # 尝试相对于当前目录查找
    #     console.print(f"[yellow]Warning: {task_prompts} not found, trying absolute path...[/yellow]")
    #     task_prompts_path = Path(os.getcwd()) / task_prompts
        
    if not task_prompts_path.exists():
        console.print(f"[bold red]Critical Error: Task prompts file not found at {task_prompts_path}[/bold red]")
        console.print("Please create 'config/pde_task_prompts.yaml' first.")
        return

    # 3. 加载 Base Config (default.yaml)
    # 这一步是为了获取 MiniSWE 的基础能力（Bash交互规则、错误处理模板等）
    default_config_path = package_dir / "config" / "default.yaml"
    if not default_config_path.exists():
        console.print(f"[bold red]Error: Default config not found at {default_config_path}[/bold red]")
        return
    
    # 读取 Base Config
    full_default_yaml = yaml.safe_load(default_config_path.read_text())
    base_agent_config = full_default_yaml.get("agent", {})

    # 构造运行时通用参数 (将被 Generator 和 Verifier 共同继承)
    runtime_common_config = {
        **base_agent_config,  # 继承 Bash 规则
        "mode": "yolo",       # 自动执行模式
        "confirm_exit": False,
        "quiet": True         # 减少控制台输出噪音
    }

    # 4. 获取题目列表
    all_cases = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    if target_case:
        all_cases = [p for p in all_cases if p.name == target_case]
        if not all_cases:
            console.print(f"[red]Case '{target_case}' not found![/red]")
            return

    console.print(f"[bold green]Found {len(all_cases)} cases to process.[/bold green]")
    console.print(f"[bold blue]Using Task Prompts from: {task_prompts_path}[/bold blue]")

    # 记录原始工作目录以便恢复
    original_cwd = os.getcwd()

    # 5. 循环处理每个 Case
    for case_folder in all_cases:
        case_id = case_folder.name
        console.rule(f"[bold magenta]Processing: {case_id}[/bold magenta]")

        # 准备结果目录
        case_output_dir = results_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # 读取题目描述
        desc_file = case_folder / "description.md"
        if not desc_file.exists():
            console.print(f"[yellow]Skipping {case_id}: description.md missing[/yellow]")
            continue
        
        # 构造完整的 Generator 初始 Prompt
        task_prompt_text = get_full_prompt(desc_file.read_text())

        # --- 核心逻辑开始 ---
        # 切换工作目录到结果文件夹，确保 Agent 生成的文件都在这里
        os.chdir(case_output_dir)
        console.print(f"📂 Working directory: {case_output_dir}")

        try:
            # 初始化模型
            model_obj = LitellmModel(model_name=model)
            
            # 初始化环境 (LocalEnvironment 默认使用 os.getcwd())
            env = LocalEnvironment()

            # 初始化编排器
            # 关键点：传入 Base Config (能力) 和 Task Prompts Path (业务)
            orchestrator = PDEOrchestrator(
                model=model_obj,
                env=env,
                base_config=runtime_common_config,
                task_prompts_path=task_prompts_path
            )

            # 运行双 Agent 流水线
            exit_status, final_result = orchestrator.run(task_prompt_text)

            if exit_status == "Submitted":
                console.print(f"[bold green]✅ Success: {case_id}[/bold green]")
            else:
                console.print(f"[bold red]❌ Failed: {case_id} (Status: {exit_status})[/bold red]")

        except Exception as e:
            console.print(f"[bold red]🔥 Exception in {case_id}: {e}[/bold red]")
            import traceback
            traceback.print_exc()
        finally:
            # 无论成功失败，切回原始目录
            os.chdir(original_cwd)

if __name__ == "__main__":
    app()