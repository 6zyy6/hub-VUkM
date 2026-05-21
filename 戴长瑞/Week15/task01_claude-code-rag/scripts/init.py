"""初始化目录"""
from app.utils import ensure_dir

def init():
    ensure_dir("./data/uploads")
    ensure_dir("./data/parsed")
    ensure_dir("./data/images")
    print("Directories initialized")

if __name__ == "__main__":
    init()