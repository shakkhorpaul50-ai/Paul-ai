import os
import subprocess

from llama_cpp import Llama

# Qwen2.5-0.5B Q3_K_M ~355MB (actual with fallback for 896 hidden, ideal 210MB)
# 24 layers, 896 hidden, Q3_K_M quantized: 988M -> 355MB (144/169 tensors fallback to IQ4_NL/q5_0)
# Fits Render Free 512MB with N_CTX=256: 355MB + runtime ~130MB + KV ~35MB = ~520MB (borderline, use N_CTX 256 default)
DEFAULT_MODEL = "models/qwen2.5-0.5b-bangla-Q3_K_M.gguf"


RELEASE_URL = "https://github.com/shakkhorpaul50-ai/Paul-ai/releases/download/v1.0/qwen2.5-0.5b-bangla-Q3_K_M.gguf"


def _download_via_curl(url: str, dest: str) -> bool:
    # Try curl first, then urllib fallback - works on Render even without git-lfs
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[model] Downloading 355MB from {url} -> {dest} ...")
    try:
        # Prefer curl (faster, shows progress, available on Render)
        result = subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", dest, url],
            timeout=300,
            check=False,
        )
        if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
            print(f"[model] curl succeeded: {os.path.getsize(dest)/1e6:.1f} MB")
            return True
        print(f"[model] curl failed code {result.returncode}, trying urllib...")
    except Exception as e:
        print(f"[model] curl exception: {e}")
    try:
        import urllib.request

        print("[model] Trying urllib fallback...")
        urllib.request.urlretrieve(url, dest)
        if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
            print(f"[model] urllib succeeded: {os.path.getsize(dest)/1e6:.1f} MB")
            return True
    except Exception as e:
        print(f"[model] urllib failed: {e}")
    return False


def _ensure_model(path: str) -> str:
    # Handle missing file or Render git-lfs pointer not fetched (<2MB + marker)
    try:
        needs_download = False
        reason = ""
        if not os.path.exists(path):
            needs_download = True
            reason = "file missing"
        else:
            size = os.path.getsize(path)
            if size < 2 * 1024 * 1024:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(512)
                        if "https://git-lfs.github.com/spec/v1" in head:
                            needs_download = True
                            reason = f"LFS pointer ({size} bytes)"
                except:
                    pass
                if not needs_download and size < 10 * 1024 * 1024:
                    needs_download = True
                    reason = f"too small ({size} bytes, expected >10MB)"
        if needs_download:
            print(f"[model] Detected {reason} for {path}, trying to fetch...")
            # 1. Try git lfs pull (works if git-lfs installed, e.g., local)
            try:
                subprocess.run(["git", "lfs", "pull"], check=False, timeout=60)
                subprocess.run(["git", "lfs", "fetch", "--all"], check=False, timeout=60)
                if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
                    print(f"[model] LFS pull succeeded: {os.path.getsize(path)/1e6:.1f} MB")
                    return path
            except Exception as e:
                print(f"[model] git lfs pull failed: {e}")
            # 2. Try GitHub Release direct download (no git-lfs needed, works on Render)
            release_url = os.environ.get("MODEL_RELEASE_URL", RELEASE_URL)
            if _download_via_curl(release_url, path):
                return path
            # 3. Try HuggingFace Hub if configured
            hf_repo = os.environ.get("MODEL_HF_REPO", "")
            if hf_repo:
                try:
                    from huggingface_hub import hf_hub_download
                    import shutil

                    print(f"[model] Downloading from HF {hf_repo}...")
                    dl = hf_hub_download(repo_id=hf_repo, filename=os.path.basename(path))
                    shutil.copy(dl, path)
                    return path
                except Exception as e:
                    print(f"[model] HF download failed: {e}")
            print(f"[model] WARNING: {path} still missing/pointer after all fetches")
        if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
            return path
    except Exception as e:
        print(f"[model] ensure error: {e}")
        import traceback

        traceback.print_exc()
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

# Paul AI identity - overrides Qwen's default "You are Qwen, created by Alibaba Cloud"
_system_prompt = (
    "You are Paul AI, a helpful Bangla AI assistant built by shakkhorpaul50-ai. "
    "You were created and trained by shakkhorpaul50-ai. "
    "When asked who built you, who created you, who made you, or 'who is your creator', "
    "always answer: you were built by shakkhorpaul50-ai. "
    "Never claim to be Qwen, Alibaba, Meta, Muse, OpenAI, or any other company. "
    "Be concise and clear. Respond in the same language the user uses."
)


class ChatModel:
    def __init__(self, model_path: str = MODEL_PATH):
        # Render Free 512MB tuned: n_ctx 128 default (355MB model needs minimal KV), n_batch 4 (was 32), mmap=True
        # 500M Q3_K_M actual 355MB + runtime ~130MB + KV ~17MB (128 ctx) = ~502MB fits 512MB
        # Set N_CTX=256 or 512 only if you upgrade Render plan beyond free
        n_ctx = int(os.environ.get("N_CTX", "128"))
        n_batch = int(os.environ.get("N_BATCH", "4"))
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
            stop=["</s>", "<|im_end|>", "<|end|>", "User:", "User"],
        )
        text = response["choices"][0]["message"]["content"] or ""
        # Identity guard: correct any Alibaba/Qwen leak from base model
        low = text.lower()
        if "alibaba" in low or "qwen" in low or "muse spark" in low:
            # If model incorrectly claims Alibaba/Qwen/Muse, override with correct identity
            if "who" in low and ("built" in low or "created" in low or "made" in low):
                return "I was built by shakkhorpaul50-ai. I am Paul AI."
        return text.strip() or "Sorry, I could not generate a response. Please try again."
