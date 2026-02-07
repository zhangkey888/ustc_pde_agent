#!/usr/bin/env python3
"""
测试脚本 - Case: reaction_diffusion_linear_basic (Fix Time Budget, Optimize Accuracy)

目标：在时间预算内，优化精度

用法：
    python test_fix_time.py --agent-script /path/to/solver.py
"""

import argparse
import sys
from pathlib import Path

# 添加pdebench到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdebench.testing.test_base import EllipticCaseTest


def main():
    parser = argparse.ArgumentParser(
        description='Test case: reaction_diffusion_linear_basic (Fix Time Budget, Optimize Accuracy)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--agent-script',
        type=Path,
        required=True,
        help='Path to agent solver script'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    # 验证agent脚本存在
    if not args.agent_script.exists():
        print(f"❌ Error: Agent script not found: {args.agent_script}")
        sys.exit(1)
    
    # 提取agent目录
    agent_dir = args.agent_script.parent.parent
    
    # 创建测试实例
    case_dir = Path(__file__).parent
    tester = EllipticCaseTest(case_dir, agent_dir=agent_dir)
    
    # 运行测试
    result = tester.run_test(
        agent_script=args.agent_script,
        test_mode='fix_time',
        timeout_sec=args.timeout
    )
    
    # 返回状态码
    if result['status'] == 'PASSED':
        print("\n✅ TEST PASSED")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
