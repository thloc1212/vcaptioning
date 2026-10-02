"""Download only the public YouCook2 files required for inference."""

from huggingface_hub import hf_hub_download


FILES = (
    ("Geppa/HiCM2", "dataset", "data/yc2/clipvitl14.pth"),
    ("Geppa/HiCM2", "dataset", "data/yc2/val.json"),
    ("Geppa/HiCM2", "dataset", "data/yc2/youcook2_asr_align_proc.pkl"),
    ("Geppa/HiCM2", "model", "presave/yc2/best_model.pth"),
)


def main():
    for repo_id, repo_type, filename in FILES:
        path = hf_hub_download(repo_id=repo_id, repo_type=repo_type,
                               filename=filename, local_dir=".")
        print(path, flush=True)


if __name__ == "__main__":
    main()
