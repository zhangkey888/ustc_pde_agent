#!/usr/bin/env python3
"""
测试脚本 - Case: stokes_no_exact_corner_force_outflow (Fix Accuracy, Optimize Speed)

目标：保证精度要求，优化速度

用法：
    python test_fix_accuracy.py --agent-script /path/to/solver.py
"""

import argparse
import sys
from pathlib import Path

# 添加pdebench到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdebench.testing.test_base import IncompressibleFlowCaseTest


def main():
    parser = argparse.ArgumentParser(
        description='Test case: stokes_no_exact_corner_force_outflow (Fix Accuracy, Optimize Speed)',
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
    tester = IncompressibleFlowCaseTest(case_dir, agent_dir=agent_dir)
    
    # 运行测试
    result = tester.run_test(
        agent_script=args.agent_script,
        test_mode='fix_accuracy',
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
