import argparse
from huggingface_hub import snapshot_download, hf_hub_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download local model files explicitly.')
    parser.add_argument('--embeddings-only', action='store_true')
    args = parser.parse_args()
    snapshot_download('sentence-transformers/all-MiniLM-L6-v2',
        allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'])
    if not args.embeddings_only:
        hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf')
    print('Model files are cached. Run python load_data.py next.')
