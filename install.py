from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="TheBloke/phi-2-GGUF",
    repo_type="model",
    local_dir="./models/phi-2-gguf",
    local_dir_use_symlinks=False
)