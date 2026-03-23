
import sys
import importlib
import inspect
import pkgutil

class InspectTool:
    def run(self, target: str) -> str:
        """
        Introspects a python module/object to find its structure or docstring.
        """
        if not target:
            return "Error: Please provide a target to inspect (e.g., 'dolfinx.geometry')."

        try:
            # 1. 尝试解析目标
            parts = target.split('.')
            obj = None
            
            # 逐步导入，处理类似 "dolfinx.geometry" 这种 submodule 情况
            # 我们先尝试导入最顶层的包
            try:
                module = importlib.import_module(parts[0])
                obj = module
            except ImportError as e:
                return f"Error: Could not import module '{parts[0]}'. Is it installed? ({str(e)})"

            # 逐层深入 (例如从 dolfinx -> geometry -> compute_collisions_points)
            for part in parts[1:]:
                if not hasattr(obj, part):
                    # 有时候 submodule 需要显式 import 才能 getattr 到
                    try:
                        # 尝试构建完整路径重新 import (例如 dolfinx.geometry)
                        current_path = obj.__name__ + "." + part
                        importlib.import_module(current_path)
                        obj = getattr(obj, part)
                    except (ImportError, AttributeError):
                        return f"Error: '{part}' not found in '{obj.__name__}'"
                else:
                    obj = getattr(obj, part)

            # 2. 根据对象类型生成报告
            output = []
            output.append(f"=== Inspection Report for '{target}' ===\n")
            output.append(f"Type: {type(obj).__name__}\n")

            # 情况 A: 如果是模块 (Module) 或 类 (Class) -> 列出成员
            if inspect.ismodule(obj) or inspect.isclass(obj):
                # 获取该对象下的所有公开成员
                members = [m for m in dir(obj) if not m.startswith('_')]
                
                # 如果是包(Package)，尝试列出子模块
                if inspect.ismodule(obj) and hasattr(obj, "__path__"):
                    for _, name, _ in pkgutil.iter_modules(obj.__path__):
                        if name not in members:
                            members.append(name)
                
                members.sort()
                
                # 为了防止输出太长，简单的排版一下
                output.append(f"--- Available Members ({len(members)}) ---")
                # 每行显示几个，看起来整洁一点
                chunk_size = 3
                for i in range(0, len(members), chunk_size):
                    output.append(", ".join(members[i:i+chunk_size]))
                
                # 如果是类，顺便显示一下类的文档
                if inspect.isclass(obj) and obj.__doc__:
                     output.append(f"\n--- Class Docstring ---\n{inspect.getdoc(obj)}")

            # 情况 B: 如果是函数 (Function) 或 方法 (Method) -> 显示签名和文档
            elif inspect.isroutine(obj):
                try:
                    sig = inspect.signature(obj)
                    output.append(f"Signature: {obj.__name__}{sig}\n")
                except ValueError:
                    output.append(f"Signature: {obj.__name__}(...)\n")
                
                doc = inspect.getdoc(obj)
                if doc:
                    output.append(f"--- Docstring ---\n{doc}")
                else:
                    output.append("--- No Docstring available ---")

            else:
                # 其他对象（如常量），直接打印值
                output.append(f"Value: {str(obj)[:500]}") # 截断防止太长

            return "\n".join(output)

        except Exception as e:
            return f"Inspect execution failed: {str(e)}"