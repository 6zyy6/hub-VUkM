@echo off
echo ============================================
echo  启动 Redis Stack 容器 (端口 6379 + 8001)
echo ============================================
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Redis Stack 启动成功!
    echo   - Redis 服务: localhost:6379
    echo   - Redis Insight: http://localhost:8001
    echo.
    echo 验证连接: docker exec redis-stack redis-cli ping
) else (
    echo.
    echo 启动失败，请检查 Docker Desktop 是否正在运行。
)
pause
