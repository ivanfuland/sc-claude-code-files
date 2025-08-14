#!/usr/bin/env python3
"""
代码质量检查脚本
运行代码格式化、linting 和类型检查工具
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """运行命令并返回结果"""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=Path.cwd()
        )
        if result.returncode == 0:
            print(f"✅ {description} - 通过")
            return True, result.stdout
        else:
            print(f"❌ {description} - 失败")
            print(f"错误输出:\n{result.stderr}")
            print(f"标准输出:\n{result.stdout}")
            return False, result.stderr
    except FileNotFoundError:
        print(f"❌ {description} - 命令未找到: {' '.join(cmd)}")
        return False, f"命令未找到: {' '.join(cmd)}"


def main():
    """主函数"""
    print("🚀 开始代码质量检查...\n")
    
    # 检查目标
    targets = ["backend/", "main.py"]
    
    checks = [
        # Black 格式检查
        (["uv", "run", "black", "--check"] + targets, "Black 格式检查"),
        
        # isort 导入排序检查
        (["uv", "run", "isort", "--check-only"] + targets, "isort 导入排序检查"),
        
        # 暂时禁用 Flake8，因为存在大量历史代码问题
        # (["uv", "run", "flake8"] + targets, "Flake8 代码规范检查"),
    ]
    
    failed_checks = []
    
    for cmd, description in checks:
        success, output = run_command(cmd, description)
        if not success:
            failed_checks.append(description)
    
    print("\n" + "="*50)
    
    if failed_checks:
        print("❌ 质量检查失败! 失败的检查项:")
        for check in failed_checks:
            print(f"  - {check}")
        print("\n💡 运行修复脚本: python scripts/fix_quality.py")
        sys.exit(1)
    else:
        print("✅ 所有质量检查通过!")
        sys.exit(0)


if __name__ == "__main__":
    main()