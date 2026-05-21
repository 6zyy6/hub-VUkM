"""
环境配置检查脚本 - 运行前执行此脚本检查依赖
"""
import os
import sys
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("多模态RAG聊天机器人 - 环境检查")
    print("=" * 60)

    issues = []

    # 1. 检查Python版本
    print(f"\n✓ Python版本: {sys.version}")
    if sys.version_info < (3, 9):
        issues.append("⚠ Python版本过低，建议3.9+")

    # 2. 检查依赖包
    required_packages = [
        'streamlit', 'kafka', 'pymilvus',
        'sentence_transformers', 'openai', 'sqlalchemy'
    ]

    print("\n检查依赖包:")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            issues.append(f"✗ 缺少依赖: {package}")
            print(f"  ✗ {package} - 未安装")

    # 3. 检查模型路径
    print("\n检查模型配置:")
    bge_path = os.getenv('BGE_MODEL_PATH', './models/bge-small-zh-v1.5')
    clip_path = os.getenv('CLIP_MODEL_PATH', './models/jina-clip-v2')

    if os.path.exists(bge_path):
        print(f"  ✓ BGE模型: {bge_path}")
    else:
        issues.append(f"⚠ BGE模型不存在: {bge_path}")
        print(f"  ✗ BGE模型不存在: {bge_path}")

    if os.path.exists(clip_path):
        print(f"  ✓ CLIP模型: {clip_path}")
    else:
        issues.append(f"⚠ CLIP模型不存在: {clip_path}")
        print(f"  ✗ CLIP模型不存在: {clip_path}")

    # 4. 检查目录
    print("\n检查目录:")
    for dir_name in ['uploads', 'processed']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✓ 创建目录: {dir_name}")
        else:
            print(f"  ✓ 目录存在: {dir_name}")

    # 5. 检查环境变量
    print("\n检查环境变量:")
    env_vars = {
        'QWEN_API_KEY': '通义千问API密钥',
        'MILVUS_URI': 'Milvus连接URI',
        'MILVUS_TOKEN': 'Milvus访问Token',
    }

    for var, desc in env_vars.items():
        if os.getenv(var):
            value = os.getenv(var)
            masked = value[:8] + '***' if len(value) > 8 else '***'
            print(f"  ✓ {desc}: {masked}")
        else:
            issues.append(f"⚠ 未设置环境变量: {var}")
            print(f"  ✗ {desc}: 未设置")

    # 6. 检查外部服务
    print("\n检查外部服务:")
    import socket

    # Kafka
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 9092))
        if result == 0:
            print("  ✓ Kafka服务: 运行中")
        else:
            issues.append("⚠ Kafka服务: 未启动 (localhost:9092)")
            print("  ✗ Kafka服务: 未启动")
        sock.close()
    except:
        issues.append("⚠ 无法检查Kafka服务")
        print("  ? Kafka服务: 检查失败")

    # MinerU
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 30000))
        if result == 0:
            print("  ✓ MinerU服务: 运行中")
        else:
            issues.append("⚠ MinerU服务: 未启动 (127.0.0.1:30000)")
            print("  ✗ MinerU服务: 未启动")
        sock.close()
    except:
        issues.append("⚠ 无法检查MinerU服务")
        print("  ? MinerU服务: 检查失败")

    # 总结
    print("\n" + "=" * 60)
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n请先解决上述问题后再运行应用！")
    else:
        print("✓ 环境检查通过！可以启动应用。")
    print("=" * 60)

    return len(issues) == 0


if __name__ == "__main__":
    check_environment()
