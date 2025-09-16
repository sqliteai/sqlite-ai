import os
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformSpecificWheel(bdist_wheel):
    """Custom bdist_wheel to force platform-specific wheel."""

    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Force platform-specific wheel
        self.root_is_pure = False

        # Set platform name from environment if provided
        plat_name = os.environ.get("PLAT_NAME")
        if plat_name:
            self.plat_name = plat_name

    def get_tag(self):
        # Force platform-specific tags with broader compatibility
        python_tag, abi_tag, platform_tag = bdist_wheel.get_tag(self)

        # Override platform tag if specified
        plat_name = os.environ.get("PLAT_NAME")
        if plat_name:
            platform_tag = plat_name

        # Use py3 for broader Python compatibility since we have pre-built binaries
        python_tag = "py3"
        abi_tag = "none"

        return python_tag, abi_tag, platform_tag


def get_platform_classifiers():
    """Get platform-specific classifiers based on PLAT_NAME environment variable."""
    classifier_map = {
        "manylinux2014_x86_64": ["Operating System :: POSIX :: Linux"],
        "manylinux2014_aarch64": ["Operating System :: POSIX :: Linux"],
        "win_amd64": ["Operating System :: Microsoft :: Windows"],
        "macosx_10_9_x86_64": ["Operating System :: MacOS"],
        "macosx_11_0_arm64": ["Operating System :: MacOS"],
    }

    plat_name = os.environ.get("PLAT_NAME")
    if plat_name and plat_name in classifier_map:
        return ["Programming Language :: Python :: 3", classifier_map[plat_name][0]]

    raise ValueError(f"Unsupported or missing PLAT_NAME: {plat_name}")


if __name__ == "__main__":
    setup(
        cmdclass={"bdist_wheel": PlatformSpecificWheel},
        classifiers=get_platform_classifiers(),
    )
