import os
import subprocess

from llama_cpp import Llama

# Qwen2.5-0.5B Q3_K_M ~355MB (actual with fallback for 896 hidden, ideal 210MB)
# 24 layers, 896 hidden, Q3_K_M quantized: 988M -> 355MB (144/169 tensors fallback to IQ4_NL/q5_0)
# Fits Render Free 512MB with N_CTX=256: 355MB + runtime ~130MB + KV ~35MB = ~520MB (borderline, use N_CTX 256 default)
DEFAULT_MODEL = "models/qwen2.5-0.5b-bangla-Q3_K_M.gguf"


def _ensure_model(path: str) -> str:
    # Handle Render git-lfs pointer not fetched (file <1MB and contains git-lfs marker)
    try:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size < 2 * 1024 * 1024:  # <2MB likely pointer
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(512)
                    if "https://git-lfs.github.com/spec/v1" in head:
                        print(f"[model] Detected LFS pointer for {path} ({size} bytes), trying git lfs pull...")
                        try:
                            subprocess.run(["git", "lfs", "pull"], check=False, timeout=120)
                            # Also try fetch
                            subprocess.run(["git", "lfs", "fetch", "--all"], check=False, timeout=120)
                            if os.path.getsize(path) > 10 * 1024 * 1024:
                                print(f"[model] LFS pull succeeded, now {os.path.getsize(path)/1e6:.1f} MB")
                                return path
                        except Exception as e:
                            print(f"[model] git lfs pull failed: {e}")
                        # Fallback: try to download via huggingface_hub if available
                        # User can set MODEL_HF_REPO env var
                        hf_repo = os.environ.get("MODEL_HF_REPO", "")
                        if hf_repo:
                            try:
                                from huggingface_hub import hf_hub_download

                                print(f"[model] Downloading from HF {hf_repo}...")
                                dl = hf_hub_download(repo_id=hf_repo, filename=os.path.basename(path))
                                import shutil

                                shutil.copy(dl, path)
                                return path
                            except Exception as e:
                                print(f"[model] HF download failed: {e}")
                        print(f"[model] WARNING: {path} is still LFS pointer, falling back to legacy if exists")
        if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
            return path
    except Exception as e:
        print(f"[model] ensure error: {e}")
    return path


# Resolve model path: env var or default, with LFS fallback and legacy SmolLM fallback
_requested = os.environ.get("MODEL_PATH", DEFAULT_MODEL)
_resolved = _ensure_model(_requested)
if os.path.exists(_resolved) and os.path.getsize(_resolved) > 10 * 1024 * 1024:
    MODEL_PATH = _resolved
elif os.path.exists(DEFAULT_MODEL) and os.path.getsize(DEFAULT_MODEL) > 10 * 1024 * 1024:
    MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL)
    _ensure_model(MODEL_PATH)
else:
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/smolLM-135m-chat-reasoning-Q4_K_M.gguf")

# Qwen system prompt Bongla-aware, concise
_system_prompt = (
    "You are a helpful Bangla AI assistant. You can have conversations, answer questions, "
    "and help solve problems step by step. Be concise and clear. Respond in the same language the user uses."
)


class ChatModel:
    def __init__(self, model_path: str = MODEL_PATH):
        # Render Free 512MB tuned: n_ctx 256 default (355MB model needs smaller KV), n_batch 8 (was 32), mmap=True
        # 500M Q3_K_M actual 355MB + runtime ~130MB + KV ~35MB (256 ctx) = ~520MB fits 512MB borderline
        # Set N_CTX=512 only if you upgrade Render plan beyond free
        n_ctx = int(os.environ.get("N_CTX", "256"))
        n_batch = int(os.environ.get("N_BATCH", "8"))
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=1,
            n_batch=n_batch,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )

    def generate(self, user_message: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": _system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "<|end|>", "User:"],
        )
        return response["choices"][0]["message"]["content"]
