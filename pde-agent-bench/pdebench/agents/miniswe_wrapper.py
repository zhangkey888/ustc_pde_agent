"""
MiniSWE Agent Wrapper for PDEBench (Final Integration)
"""
import sys
import re
import yaml
import os
from pathlib import Path
from typing import Dict, Any

# 引入 PDEBench 基类
from .base_agent import BaseAgent, AgentResponse

try:
    from minisweagent.agents.interactive import InteractiveAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent import package_dir 
except ImportError as e:
    raise ImportError(f"无法导入 minisweagent: {e}")

class MiniSWEWrapper(BaseAgent):
    def _setup(self):
        # 1. 基础配置
        self.model_name = self.config.get('model', 'gpt-4o')
        self.max_turns = int(self.config.get('max_turns', 20)) # 给多点轮数方便Debug
        
        # 2. 设置结果存储路径 (对应你脚本里的 results_dir)
        # 默认路径硬编码为你提供的路径，也可以在 configs/miniswepde.json 里改
        default_root = "/data5/store1/zky/pde-agent-bench/results/miniswepde"
        self.results_root = Path(self.config.get('results_root', default_root))
        
        # 3. 加载 MiniSWE 默认配置 (填补 Pydantic 校验所需的 templates)
        self.agent_config = self._load_default_miniswe_config()
        
        # 4. 覆盖运行参数
        self.agent_config.update({
            "mode": "yolo",        # 全自动模式
            "quiet": True,         # 减少干扰输出
            "return_history": True,# 必须开启，否则拿不到日志
            "confirm_exit": False  # 禁止交互确认，防止卡死
        })

        print(f"🔧 Initializing MiniSWE Wrapper")
        print(f"📂 Persistent Workspace: {self.results_root}")

    def _load_default_miniswe_config(self) -> Dict[str, Any]:
        """从包安装目录加载 default.yaml"""
        try:
            config_path = package_dir / "config" / "default.yaml"
            if not config_path.exists(): return {}
            return yaml.safe_load(config_path.read_text()).get("agent", {})
        except Exception:
            return {}

    def generate_solution(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """
        核心执行逻辑
        """
        # 1. 准备工作目录: /results/miniswepde/{case_id}
        case_id = context.get('case_id')
        case_workspace = self.results_root / case_id
        case_workspace.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 MiniSWE starts for case: {case_id}")
        
        # 记录原始目录，确保跑完能切回来
        original_cwd = os.getcwd()

        try:
            # === 2. 物理隔离：强制切换当前工作目录 ===
            # 这是解决 path 问题最彻底的方法
            os.chdir(case_workspace)
            # print(f"  📂 CWD switched to: {os.getcwd()}")

            # 3. 初始化环境
            # 既然已经 chdir 了，workspace_dir 用 "." 即可
            env = LocalEnvironment(workspace_dir=".")
            model = LitellmModel(model_name=self.model_name)
            
            agent = InteractiveAgent(
                model=model, 
                env=env, 
                **self.agent_config
            )

            # 4. 构造 Prompt
            # 注意：传入的 `prompt` 已经是 PDEBench 生成的包含 `def solve` 要求的标准 Prompt
            # 我们只需要追加“行动指南”，告诉 Agent 要写文件、要运行测试
            
            action_instructions = (
                "\n\n"
                "===========================================================\n"
                "AGENT ACTION GUIDELINES:\n"
                "1. **Write Code**: Save your solution to `solver.py` in the current directory.\n"
                "2. **Self-Correction**: You MUST execute `python solver.py` to check for syntax errors or library issues.\n"
                "3. **Fix Errors**: If it fails, analyze the error and rewrite `solver.py`.\n"
                "4. **Finalize**: Ensure `solver.py` exists before finishing.\n"
                "   (Note: Do NOT write output to subdirectories like 'src/', keep it flat.)\n"
                "===========================================================\n"
            )
            
            full_prompt = prompt + action_instructions
            
            # 5. 运行 Agent
            # 捕获可能的 SystemExit
            try:
                result_tuple = agent.run(full_prompt)
            except SystemExit:
                result_tuple = ("EXIT", [])

            # 解析返回结果
            if isinstance(result_tuple, tuple):
                result_history = result_tuple[1]
            else:
                result_history = result_tuple

            # 6. 提取代码 (在当前目录下找)
            final_code = ""
            solver_file = Path("solver.py") # 相对路径，因为我们在 CWD 里
            
            # A. 优先读取磁盘文件
            if solver_file.exists():
                print(f"  📄 Found solver.py on disk.")
                final_code = solver_file.read_text()
            else:
                # B. 递归搜索（防止它建了子目录）
                print(f"  ⚠️ solver.py missing in root. Searching recursively...")
                found = list(Path(".").rglob("solver.py"))
                if found:
                    target = found[0]
                    print(f"  📄 Found at {target}")
                    final_code = target.read_text()
                else:
                    # C. 从历史提取
                    print(f"  🕵️ Extracting from history...")
                    final_code = self._extract_code_from_history(result_history)
                    # 如果提取到了，帮它写进文件，方便后续离线评测
                    if final_code:
                        solver_file.write_text(final_code)

            # 7. 返回结果
            if not final_code:
                return AgentResponse(
                    success=False,
                    code="",
                    raw_response=str(result_history),
                    agent_name=self.agent_name,
                    error="No code produced."
                )

            return AgentResponse(
                success=True,
                code=final_code,
                raw_response=str(result_history),
                agent_name=self.agent_name
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AgentResponse(
                success=False,
                code="",
                raw_response=str(e),
                agent_name=self.agent_name,
                error=str(e)
            )
        
        finally:
            # === 8. 必须还原工作目录 ===
            os.chdir(original_cwd)
            # print(f"  🔙 CWD restored.")

    def _extract_code_from_history(self, result_obj) -> str:
        content = str(result_obj)
        # 1. Markdown
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_blocks: return code_blocks[-1]
        
        # 2. Cat 命令 (MiniSWE 常用)
        bash_pattern = r'cat\s+<<\s*([A-Za-z0-9_]+)\s*>\s*solver\.py\s*\n(.*?)\n\s*\1'
        bash_matches = re.findall(bash_pattern, content, re.DOTALL)
        if bash_matches: return bash_matches[-1][1]
        
        # 3. 兜底
        if "def solve" in content: return content
        return ""
    
    def cleanup(self):
        pass