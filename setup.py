"""
Setup script for Face Swap Super
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-swap-super",
    version="1.0.0",
    author="Face Swap Super Team",
    author_email="support@faceswapsuper.com",
    description="A cutting-edge, high-performance streaming face swap system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-swap-super",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-swap-super=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords="face-swap, deepfake, ai, computer-vision, real-time, streaming",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/face-swap-super/issues",
        "Source": "https://github.com/yourusername/face-swap-super",
        "Documentation": "https://docs.faceswapsuper.com",
        "Funding": "https://github.com/sponsors/yourusername",
    },
)