import sys
import zipfile
import requests
from pathlib import Path
import shutil


# == USAGE ==
# python3 download_artifacts.py PLATFORM VERSION
#   eg: python3 download_artifacts.py linux_x86_64 "0.5.9"

REPO = "sqliteai/sqlite-ai"
RELEASE_URL = f"https://github.com/{REPO}/releases/download"

# Map Python plat_name to artifact names
ARTIFACTS = {
    "manylinux2014_x86_64": ["ai-linux-cpu-x86_64", "ai-linux-gpu-x86_64"],
    "manylinux2014_aarch64": [
        "ai-linux-cpu-arm64",
        "ai-linux-gpu-arm64",
    ],
    "win_amd64": ["ai-windows-cpu-x86_64", "ai-windows-gpu-x86_64"],
    "macosx_10_9_x86_64": ["ai-macos"],
    "macosx_11_0_arm64": ["ai-macos"],
}

BINARY_NAME = {
    "manylinux2014_x86_64": "ai.so",
    "manylinux2014_aarch64": "ai.so",
    "win_amd64": "ai.dll",
    "macosx_10_9_x86_64": "ai.dylib",
    "macosx_11_0_arm64": "ai.dylib",
}

BINARIES_DIR = Path(__file__).parent / "src/sqliteai/binaries"


def download_and_extract(artifact_name, bin_name, version):
    artifact = f"{artifact_name}-{version}.zip"
    url = f"{RELEASE_URL}/{version}/{artifact}"
    print(f"Downloading {url}")

    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to download {artifact}: {r.status_code}")
        sys.exit(1)

    zip_path = BINARIES_DIR / artifact
    with open(zip_path, "wb") as f:
        f.write(r.content)

    subdir = "gpu" if "gpu" in artifact_name else "cpu"
    out_dir = BINARIES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith(bin_name):
                zip_ref.extract(member, out_dir)

                # Move to expected name/location
                src = out_dir / member
                dst = out_dir / bin_name
                src.rename(dst)

                print(f"Extracted {dst}")

    zip_path.unlink()


def main():
    version = None
    platform = None
    if len(sys.argv) == 3:
        platform = sys.argv[1].lower()
        version = sys.argv[2]

    if not version or not platform:
        print(
            'Error: Version is not specified.\nUsage: \n   python3 download_artifacts.py linux_x86_64 "0.5.9"'
        )
        sys.exit(1)

    print(BINARIES_DIR)
    if BINARIES_DIR.exists():
        shutil.rmtree(BINARIES_DIR)
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)

    platform_artifacts = ARTIFACTS.get(platform, [])
    if not platform_artifacts:
        print(f"Error: Unknown platform '{platform}'")
        sys.exit(1)

    for artifact_name in platform_artifacts:
        download_and_extract(artifact_name, BINARY_NAME[platform], version)


if __name__ == "__main__":
    main()
