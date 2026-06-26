from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements():
    requirements_path = ROOT / "requirements.txt"
    requirements = []

    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        requirements.append(line)

    return requirements


setup(
    name="GPKoopman",
    version="0.1.0",
    description=(
        "Inverted Gaussian Process optimization based Koopman operator "
        "discovery tools."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["GPKoopman", "GPKoopman.*"]),
    install_requires=read_requirements(),
    python_requires=">=3.12",
)
