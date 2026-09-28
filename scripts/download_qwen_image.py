"""
Download script for Qwen-Image 2.1 models into ComfyUI-Shared models directory.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

TARGET_DIR = Path("/Users/leandrobocon/ComfyUI-Shared/models")
REPO_ID = "Comfy-Org/Qwen-Image-2.1"

FILES_TO_DOWNLOAD = [
    ("vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
    ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models"),
    ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
]

def main():
    print("=" * 60)
    print("🚀 Baixando modelos do Qwen-Image 2.1 para ComfyUI")
    print("=" * 60)
    
    for filename, category in FILES_TO_DOWNLOAD:
        dest = TARGET_DIR / category / Path(filename).name
        if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
            print(f"✅ Já existe: {dest.name} ({dest.stat().st_size / (1024**3):.2f} GB)")
            continue
        
        print(f"\n📥 Baixando {filename}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(TARGET_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"✅ Salvo em: {downloaded}")
        except Exception as e:
            print(f"❌ Erro ao baixar {filename}: {e}", file=sys.stderr)
            sys.exit(1)
            
    print("\n🎉 Todos os modelos do Qwen-Image 2.1 foram baixados com sucesso!")

if __name__ == "__main__":
    main()
