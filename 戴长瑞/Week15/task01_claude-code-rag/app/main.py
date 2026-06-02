"""FastAPI 应用入口"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_config
from app.core.factory import initialize_factories

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Initializing Claude Code RAG...")
    initialize_factories()
    logger.info("All services initialized")
    yield
    # 关闭时
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    config = get_config()

    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description="多模态 RAG 系统 - 支持图文混合 PDF 解析、多模态检索与问答",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(api_router, prefix="/api/v1")

    # 健康检查
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": config.app_name,
            "version": config.app_version
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )