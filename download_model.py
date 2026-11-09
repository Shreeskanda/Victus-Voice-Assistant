from huggingface_hub import hf_hub_download
import os

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Starting TinyLlama download...")
print("This may take a few minutes.\n")

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    local_dir=MODEL_DIR
)

print("\nDownload completed successfully!")
print("Model location:", model_path)
