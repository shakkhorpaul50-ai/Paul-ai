"""
Quantize the fine-tuned model to INT4 GGUF format.

After training on Kaggle, run this to export the model:
  1. Merge LoRA adapter into base model
  2. Convert to GGUF INT4 (Q4_K_M)

Requirements on Kaggle:
  pip install llama-cpp-python huggingface_hub
"""

import argparse
import os
import subprocess
import sys

LORA_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen500M-bangla-lora")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_NAME = "qwen2.5-0.5b-bangla-Q3_K_M.gguf"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# Render Free 512MB: Q3_K_M ~210MB fits, Q4_K_M ~340MB would OOM


def merge_and_export(args):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("Install dependencies: pip install transformers peft")
        sys.exit(1)

    base_model_name = args.base_model or BASE_MODEL_NAME
    lora_dir = args.lora_dir or LORA_DIR
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()

    merged_dir = os.path.join(OUTPUT_DIR, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("Merged model saved. Now converting to GGUF...")
    convert_to_gguf(merged_dir, args.qtype)


def convert_to_gguf(merged_dir: str, qtype: str = "q3_k_m"):
    # qtype: q3_k_m (Render Free ~210MB), q4_k_m (~340MB would OOM on 512MB), q8_0, f16
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    # Allow custom name per qtype
    if qtype != "q3_k_m":
        base, _ = os.path.splitext(OUTPUT_NAME)
        output_path = os.path.join(OUTPUT_DIR, f"{base.split('-Q')[0]}-{qtype.upper()}.gguf")

    llama_cpp_dir = os.path.join(OUTPUT_DIR, "llama.cpp")
    print("Cloning llama.cpp for conversion tools...")
    if not os.path.exists(llama_cpp_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
            check=True,
        )
    else:
        # Ensure built for quantize
        print("llama.cpp already exists, skipping clone")

    print("Installing conversion requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         os.path.join(llama_cpp_dir, "requirements.txt")],
        check=True,
    )

    # Step 1: HF -> F16 GGUF (convert_hf_to_gguf.py only supports f32/f16/bf16/q8_0 for this branch)
    # q3_k_m/q4_k_m require second step via llama-quantize
    if qtype.lower() in ("q3_k_m", "q4_k_m", "q4_k_s", "q5_k_m"):
        f16_path = os.path.join(OUTPUT_DIR, "_tmp_f16.gguf")
        print(f"Step 1/2: Converting HF to F16 GGUF -> {f16_path}")
        subprocess.run(
            [sys.executable, os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
             merged_dir, "--outfile", f16_path, "--outtype", "f16"],
            check=True,
        )
        print("Building llama-quantize...")
        build_dir = os.path.join(llama_cpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", "-B", build_dir, "-S", llama_cpp_dir, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"], check=True)
        subprocess.run(["cmake", "--build", build_dir, "--config", "Release", "-j", "4"], check=True)

        quantize_bin = os.path.join(build_dir, "bin", "llama-quantize")
        if os.name == "nt":
            quantize_bin += ".exe"
            # Windows build may place binary differently
            alt = os.path.join(build_dir, "bin", "Release", "llama-quantize.exe")
            if os.path.exists(alt):
                quantize_bin = alt
        # Fallback search
        if not os.path.exists(quantize_bin):
            import glob as _glob
            cand = _glob.glob(os.path.join(llama_cpp_dir, "**", "llama-quantize*"), recursive=True)
            cand = [c for c in cand if c.endswith(".exe") or "llama-quantize" in os.path.basename(c)]
            if cand:
                quantize_bin = cand[0]

        print(f"Step 2/2: Quantizing F16 -> {qtype.upper()} -> {output_path}")
        print(f"  using {quantize_bin}")
        subprocess.run([quantize_bin, f16_path, output_path, qtype.upper()], check=True)
        # Cleanup tmp
        if os.path.exists(f16_path):
            os.remove(f16_path)
    else:
        print(f"Converting directly to {qtype} -> {output_path}")
        subprocess.run(
            [sys.executable, os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
             merged_dir, "--outfile", output_path, "--outtype", qtype],
            check=True,
        )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done. GGUF model saved to: {output_path} ({size_mb:.1f} MB)")
    if size_mb > 400:
        print("WARNING: >400MB may OOM on Render Free 512MB. Use q3_k_m (~210MB) for Render Free.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and quantize to GGUF")
    parser.add_argument("--base_model", type=str, default=None, help="Base HF model")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA adapter dir")
    parser.add_argument("--qtype", type=str, default="q3_k_m", choices=["q3_k_m", "q4_k_m", "q4_k_s", "q5_k_m", "q8_0", "f16", "q3_k_s", "q2_k"], help="GGUF quant type (q3_k_m fits Render Free 512MB)")
    args = parser.parse_args()
    merge_and_export(args)
