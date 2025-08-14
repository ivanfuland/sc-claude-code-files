#!/usr/bin/env python3
"""
代码质量修复脚本
自动修复格式和导入排序问题
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """运行命令并返回结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=Path.cwd()
        )
        if result.returncode == 0:
            print(f"✅ {description} - 完成")
            return True, result.stdout
        else:
            print(f"❌ {description} - 失败")
            print(f"错误输出:\n{result.stderr}")
            return False, result.stderr
    except FileNotFoundError:
        print(f"❌ {description} - 命令未找到: {' '.join(cmd)}")
        return False, f"命令未找到: {' '.join(cmd)}"


def main():
    """主函数"""
    print("🛠️  开始自动修复代码质量问题...\n")
    
    # 检查目标
    targets = ["backend/", "main.py"]
    
    fixes = [
        # Black 格式化
        (["uv", "run", "black"] + targets, "Black 代码格式化"),
        
        # isort 导入排序
        (["uv", "run", "isort"] + targets, "isort 导入排序"),
    ]
    
    failed_fixes = []
    
    for cmd, description in fixes:
        success, output = run_command(cmd, description)
        if not success:
            failed_fixes.append(description)
        else:
            if output.strip():
                print(f"输出: {output}")
    
    print("\n" + "="*50)
    
    if failed_fixes:
        print("❌ 部分修复失败! 失败的修复项:")
        for fix in failed_fixes:
            print(f"  - {fix}")
        print("\n💡 请手动检查并修复剩余问题")
        sys.exit(1)
    else:
        print("✅ 所有质量问题已修复!")
        print("\n🔍 建议运行质量检查确认: python scripts/quality_check.py")
        sys.exit(0)


if __name__ == "__main__":
    main()