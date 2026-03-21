
"""
MiniSWE Agent Wrapper for PDEBench (Dual-Agent Orchestrator Integration)
"""
import sys
import re
import yaml
import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Dict, Any

# 引入 PDEBench 基类
from .base_agent import BaseAgent, AgentResponse

try:
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent import package_dir 
    
    # === 尝试导入 Orchestrator ===
    try:
        from minisweagent.agents.pde_orchestrator import PDEOrchestrator
    except ImportError:
        sys.path.append(os.path.join(package_dir, "..")) # 尝试上一级
        from pde_orchestrator import PDEOrchestrator
        
except ImportError as e:
    raise ImportError(f"无法导入 minisweagent 组件: {e}")

class MiniSWEWrapper(BaseAgent):
    def _setup(self):
        # 1. 基础配置
        self.model_name = self.config.get('model', 'gpt-4o')
        
        # === 🌟 终极修改 1：提取 API Base 并强制写入双重环境变量 ===
        self.api_base = self.config.get('api_base') or self.config.get('base_url')
        if self.api_base:
            # 确保规范的 /v1 结尾
            if not self.api_base.endswith("/v1") and not self.api_base.endswith("/v1/"):
                self.api_base = self.api_base.rstrip("/") + "/v1"
            
            # 暴力写入环境变量，绕过框架的参数丢失问题
            os.environ["OPENAI_API_BASE"] = self.api_base
            os.environ["OPENAI_BASE_URL"] = self.api_base
            print(f"🌐 Global API Base set to: {self.api_base}")

        # === 🌟 终极修改 2：写入 API Key ===
        self.api_key = self.config.get('api_key') or self.config.get('key')
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
            print(f"🔑 Global API Key loaded.")
        # 2. 设置结果存储路径
        default_root = "/data5/store1/zky/ustc_pde_agent/pde-agent-bench/results/ustcpdeagent"

        self.results_root = Path(self.config.get('results_root', default_root))
        
        # 3. 加载基础配置 (default.yaml)
        self.default_config_path = package_dir / "config" / "default.yaml"
        if not self.default_config_path.exists():
            raise FileNotFoundError(f"Default config not found at {self.default_config_path}")
        
        self.full_default_yaml = yaml.safe_load(self.default_config_path.read_text())
        self.base_agent_config = self.full_default_yaml.get("agent", {})

        # 4. 加载业务 Prompt (pde_task_prompts.yaml)
        self.task_prompts_path = package_dir / "config" / "pde_task_prompts.yaml"
        if not self.task_prompts_path.exists():
             raise FileNotFoundError(f"Task prompts not found at {self.task_prompts_path}")

        # 5. 构造运行时参数 (Override)
        self.runtime_config = {
            **self.base_agent_config,
            "mode": "yolo",        # 全自动模式
            "quiet": True,         # 减少干扰输出
            "return_history": True,# 必须开启
            "confirm_exit": False  # 禁止交互确认
        }

        print(f"🔧 Initializing MiniSWE Dual-Agent Wrapper")
        if self.api_base:
            print(f"🌐 Using custom API Base URL: {self.api_base}")
        print(f"📂 Persistent Workspace: {self.results_root}")
        print(f"📜 Prompts: {self.task_prompts_path}")

    def generate_solution(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """
        执行双 Agent 流程
        """
        # 1. 准备工作目录
        raw_case_id = str(context.get('case_id', 'default_case'))
        
        # === 新增：URL 兼容模式 ===
        # 如果 case_id 是 URL（包含 http, // 或特殊字符），将其替换为安全的文件系统目录名
        safe_case_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_case_id)
        # 去掉首尾可能多余的下划线，避免目录名太丑
        safe_case_id = safe_case_id.strip("_") 
        
        case_workspace = self.results_root / safe_case_id
        case_workspace.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 MiniSWE (Dual) starts for case: {raw_case_id}")
        if raw_case_id != safe_case_id:
            print(f"📁 URL mapped to safe workspace: {safe_case_id}")
        
        original_cwd = os.getcwd()

        try:
            # === 2. 物理隔离：切换 CWD ===
            os.chdir(case_workspace)

           # 3. 初始化环境和模型
            env = LocalEnvironment(workspace_dir=".")
            
            # 兼容处理模型名称 (特别是 DeepSeek/Azure)
            # 构建 model_kwargs，传递 api_key 和 base_url 给 litellm
            model_kwargs = {}
            if self.config.get('api_key'):
                model_kwargs['api_key'] = self.config['api_key']
            if self.config.get('base_url'):
                model_kwargs['api_base'] = self.config['base_url']  # litellm 使用 api_base
            if self.config.get('temperature'):
                model_kwargs['temperature'] = self.config['temperature']
            if self.config.get('max_tokens'):
                model_kwargs['max_tokens'] = self.config['max_tokens']
            
            model_obj = LitellmModel(
                model_name=self.model_name, 
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors"  # 忽略模型价格未注册的错误
            )
            
            # 4. 初始化 Orchestrator
            orchestrator = PDEOrchestrator(
                model=model_obj,
                env=env,
                base_config=self.runtime_config,
                task_prompts_path=self.task_prompts_path
            )

            # 5. 构造 Full Prompt
            action_instructions = (
                "\n\n"
                "===========================================================\n"
                "CRITICAL INSTRUCTION:\n"
                "You are the GENERATOR agent.\n"
                "1. Write the solution to `solver.py`.\n"
                "2. Run it to verify correctness.\n"
                "3. Ensure `solution.npz` and `meta.json` are generated.\n"
                "===========================================================\n"
            )
            full_prompt = prompt + action_instructions
            
            # 6. 运行双 Agent 流水线
            try:
                exit_status, result_history = orchestrator.run(full_prompt)
            except SystemExit:
                exit_status, result_history = "EXIT", "SystemExit triggered"
            except Exception as e:
                import traceback
                traceback.print_exc()
                exit_status, result_history = "ERROR", str(e)

            # 7. 提取最终代码
            final_code = ""
            solver_file = Path("solver.py")
            
            if solver_file.exists():
                print(f"  📄 Found solver.py on disk (Verified Version).")
                final_code = solver_file.read_text()
            else:
                print(f"  ⚠️ solver.py missing. Attempting history extraction...")
                final_code = self._extract_code_from_history(result_history)
                if final_code:
                    solver_file.write_text(final_code)

            # 8. 构造返回对象
            if not final_code:
                return AgentResponse(
                    success=False,
                    code="",
                    raw_response=str(result_history),
                    agent_name=self.agent_name,
                    error=f"No code produced. Status: {exit_status}"
                )

            return AgentResponse(
                success=(exit_status == "Submitted"), 
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
            # === 9. 还原 CWD ===
            os.chdir(original_cwd)

    def _extract_code_from_history(self, result_obj) -> str:
        """从字符串日志中提取代码 (兜底策略)"""
        content = str(result_obj)
        # 1. Cat 命令 (最常见)
        bash_pattern = r'cat\s+<<\s*([A-Za-z0-9_]+)\s*>\s*solver\.py\s*\n(.*?)\n\s*\1'
        bash_matches = re.findall(bash_pattern, content, re.DOTALL)
        if bash_matches: return bash_matches[-1][1]
        
        # 2. Markdown 代码块
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_blocks: return code_blocks[-1]

        return ""

    def cleanup(self):
        pass