from logger_setup import setup_logging
setup_logging()

from config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        # log_level=settings.log_level
        log_config=None
    ) 