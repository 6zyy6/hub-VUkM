from fastapi import FastAPI
from app.routers import document, chat
from app.models.db_models import init_db
from app.config import config
from app.services.milvus_service import milvus_client

app = FastAPI(title="Multimodal RAG API", version="1.0")

@app.on_event("startup")
def on_startup():
    init_db(config.DATABASE_URL)
    milvus_client.connect()

app.include_router(document.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
