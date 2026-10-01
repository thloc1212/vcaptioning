"""Stream only YouCook2 UniVL features from the official 4 GB archive.

No compressed copy is retained. The stream still transfers the full archive.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/datasets/Exclibur/dibs-feature/resolve/main/yc2_feature.tar.xz?download=true"
PREFIXES = ("yc2/UniVL_features/UniVL_visual/", "yc2/UniVL_features/UniVL_text/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="data/features")
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with urllib.request.urlopen(URL, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|xz") as archive:
            for member in archive:
                name = member.name.lstrip("./")
                if not member.isfile() or not name.startswith(PREFIXES):
                    continue
                target = (destination / name).resolve()
                if destination not in target.parents:
                    raise ValueError(f"unsafe archive path: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                count += 1
                if count % 500 == 0:
                    print(f"extracted {count} UniVL files", flush=True)
    if not count:
        raise RuntimeError("no UniVL files found; check archive layout")
    print(f"extracted {count} files to {destination}")


if __name__ == "__main__":
    main()
