"""Install the bundled VLMEvalKit and VisionPulse model implementations."""
from pathlib import Path
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="visionpulse-vlmeval",
    version="0.1.0",
    description="VisionPulse: Dynamic Visual Sparsity for Efficient Multimodal Reasoning",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    # Some bundled evaluator modules are namespace packages without __init__.py.
    packages=find_namespace_packages(
        include=["vlmeval", "vlmeval.*", "visionpulse", "visionpulse.*"],
        exclude=["*.__pycache__", "*.__pycache__.*"],
    ),
    include_package_data=True,
    license_files=["LICENSE", "docs/THIRD_PARTY_NOTICES.md"],
    install_requires=requirements,
    entry_points={"console_scripts": ["vlmutil=vlmeval:cli"]},
)
