#!/usr/bin/env python3
"""
VLLM LoRA管理脚本
支持动态加载和卸载LoRA适配器
"""

import argparse
import requests
import json
import sys
from typing import Optional

def load_lora(port: int, lora_name: str, lora_path: str):
    """加载LoRA适配器"""
    url = f"http://localhost:{port}/v1/load_lora_adapter"
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print(f"✅ LoRA '{lora_name}' 加载成功")
        print(f"响应: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务 (端口 {port})，请确保VLLM服务正在运行")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ 加载LoRA失败: HTTP {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        sys.exit(1)

def unload_lora(port: int, lora_name: str):
    """卸载LoRA适配器"""
    url = f"http://localhost:{port}/v1/unload_lora_adapter"
    payload = {
        "lora_name": lora_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print(f"✅ LoRA '{lora_name}' 卸载成功")
        print(f"响应: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务 (端口 {port})，请确保VLLM服务正在运行")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ 卸载LoRA失败: HTTP {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="VLLM LoRA适配器管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --action load --port 17862 --lora-name test_lora --lora-path ./lora_adapters/test
  %(prog)s --action unload --port 8000 --lora-name test_lora
        """
    )
    
    parser.add_argument(
        "--action", "-a",
        required=True,
        choices=["load", "unload"],
        help="执行的操作: load (加载) 或 unload (卸载)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        required=True,
        help="VLLM服务端口号"
    )
    
    parser.add_argument(
        "--lora-name", "-n",
        required=True,
        help="LoRA适配器名称"
    )
    
    parser.add_argument(
        "--lora-path", "-l",
        help="LoRA适配器路径 (仅加载时需要)"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.action == "load" and not args.lora_path:
        parser.error("加载LoRA时需要提供 --lora-path 参数")
    
    # 执行相应操作
    if args.action == "load":
        load_lora(args.port, args.lora_name, args.lora_path)
    else:
        unload_lora(args.port, args.lora_name)

if __name__ == "__main__":
    main()
