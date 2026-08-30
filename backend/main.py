import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from model import ChatModel, MODEL_PATH as MODEL_FILE
from database import Database
from schemas import ChatRequest, ChatResponse, ConversationHistory, Message

model: ChatModel | None = None
db: Database | None = None


import threading

_model_lock = threading.Lock()
_model_loading = False

def _load_model_background():
    global model, _model_loading
    with _model_lock:
        if model is not None or _model_loading:
            return
        _model_loading = True
    try:
        print("[lifespan] Loading ChatModel in background (355MB Q3_K_M, 24 layers)...")
        m = ChatModel()
        with _model_lock:
            model = m
            print(f"[lifespan] ChatModel loaded: {model.llm}")
    except Exception as e:
        print(f"[lifespan] ChatModel failed: {e}")
        import traceback

        traceback.print_exc()
        with _model_lock:
            model = None
    finally:
        with _model_lock:
            _model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    # Start DB immediately (fast, in-memory fallback if DATABASE_URL missing)
    try:
        print("[lifespan] Connecting Database...")
        db = Database()
        print("[lifespan] Database ready")
    except Exception as e:
        print(f"[lifespan] Database failed: {e}")
        import traceback

        traceback.print_exc()
        db = None
    # Load model in background thread so Render health check passes within 10s (avoids 502)
    # Render free 512MB: 355MB model + 130MB runtime + 17MB KV (N_CTX 128) = ~502MB
    threading.Thread(target=_load_model_background, daemon=True).start()
    print("[lifespan] Model load started in background, app ready for /health")
    yield


app = FastAPI(title="AI Chat", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global model
    # Lazy load: if model not yet loaded, try to load now (first request after cold start)
    if model is None:
        # If background thread is still loading, inform client to retry
        if _model_loading:
            raise HTTPException(status_code=503, detail="Model is loading (355MB Q3_K_M, 24 layers) - please retry in 15s (check /health)")
        # Try to load synchronously if background failed
        try:
            print("[chat] Model not loaded, attempting on-demand load...")
            with _model_lock:
                if model is None:
                    model = ChatModel()
                    print(f"[chat] On-demand model loaded: {model.llm}")
        except Exception as e:
            print(f"[chat] On-demand load failed: {e}")
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Model not loaded - {e} (check /health, MODEL_PATH={os.environ.get('MODEL_PATH', MODEL_FILE)})")
    if db is None:
        # In-memory fallback already in Database, so this should not happen
        raise HTTPException(status_code=503, detail="Database not ready - check DATABASE_URL")
    conv_id = req.conversation_id
    if not conv_id:
        conv_id = db.create_conversation()

    try:
        history = db.get_history(conv_id)
    except Exception as e:
        print(f"[chat] get_history failed: {e}")
        history = []
    history_dicts = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        response_text = model.generate(req.message, history_dicts)
    except Exception as e:
        print(f"[chat] generate failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model generate failed: {e}")

    try:
        db.save_message(conv_id, "user", req.message)
        db.save_message(conv_id, "assistant", response_text)
    except Exception as e:
        print(f"[chat] db save failed: {e}")

    return ChatResponse(response=response_text, conversation_id=conv_id)


@app.get("/health")
def health():
    mpath = os.environ.get("MODEL_PATH", MODEL_FILE)
    exists = os.path.exists(mpath)
    size_mb = round(os.path.getsize(mpath)/1e6,1) if exists else 0
    # LFS pointer is <2MB and contains git-lfs marker
    is_pointer = False
    if exists and size_mb < 2:
        try:
            with open(mpath, "r", encoding="utf-8", errors="ignore") as f:
                if "https://git-lfs.github.com/spec/v1" in f.read(512):
                    is_pointer = True
        except:
            pass
    return {
        "ok": model is not None and exists and size_mb > 10,
        "loading": _model_loading,
        "model": mpath,
        "model_exists": exists,
        "model_size_mb": size_mb,
        "is_lfs_pointer": is_pointer,
        "db": db is not None and getattr(db, 'url', None) is not None,
        "db_mode": "postgres" if db and getattr(db, 'url', None) else "memory",
    }


@app.get("/api/history/{conversation_id}", response_model=ConversationHistory)
def history(conversation_id: str):
    rows = db.get_history(conversation_id)
    messages = [Message(role=r["role"], content=r["content"]) for r in rows]
    return ConversationHistory(conversation_id=conversation_id, messages=messages)


@app.delete("/api/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    db.delete_conversation(conversation_id)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse("../frontend/index.html")


app.mount("/static", StaticFiles(directory="../frontend"), name="static")
