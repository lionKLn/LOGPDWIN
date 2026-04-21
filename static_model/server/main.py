# ==========================================
# 启动入口
# ==========================================

import uvicorn

def main():
    """
    启动 FastAPI 服务
    """

    uvicorn.run(
        "server.app:app",   # 模块路径
        host="0.0.0.0",
        port=8000,
        reload=True         # 开发模式自动重启
    )


if __name__ == "__main__":
    main()