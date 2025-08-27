import setuptools
import toml
import os
import sys

usage = """
Usage: python setup.py bdist_wheel --plat-name <platform>
The PACKAGE_VERSION environment variable must be set to the desired version.

Example:
  PACKAGE_VERSION=0.5.9 python setup.py bdist_wheel --plat-name linux_x86_64
"""

with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

project = pyproject["project"]

# Get version from environment or default
version = os.environ.get("PACKAGE_VERSION", "")
if not version:
    print("PACKAGE_VERSION environment variable is not set.")
    print(usage)
    sys.exit(1)

# Get Python platform name from --plat-name argument
plat_name = None
for i, arg in enumerate(sys.argv):
    if arg == "--plat-name" and i + 1 < len(sys.argv):
        plat_name = sys.argv[i + 1]
        break

if not plat_name:
    print("Error: --plat-name argument is required")
    print(usage)
    sys.exit(1)

# Map plat_name to classifier
classifier_map = {
    "manylinux2014_x86_64": "Operating System :: POSIX :: Linux",
    "manylinux2014_aarch64": "Operating System :: POSIX :: Linux",
    "win_amd64": "Operating System :: Microsoft :: Windows",
    "macosx_10_9_x86_64": "Operating System :: MacOS",
    "macosx_11_0_arm64": "Operating System :: MacOS",
}

classifier = classifier_map.get(plat_name)
if not classifier:
    print(f"Unknown plat_name: {plat_name}")
    sys.exit(1)

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name=project["name"],
    version=version,
    description=project.get("description", ""),
    author=project["authors"][0]["name"] if project.get("authors") else "",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=project["urls"]["Homepage"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=project.get("requires-python", ">=3"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        classifier
    ],
)