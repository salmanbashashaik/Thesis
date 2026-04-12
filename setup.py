from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aldm-mri-synthesis",
    version="1.0.0",
    author="Shaik Salman Basha",
    author_email="your.email@unb.ca",
    description="Anatomically-conditioned Latent Diffusion Model for Few-Shot 3D Glioma MRI Synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aldm-mri-synthesis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aldm-preprocess=scripts.preprocess_data:main",
            "aldm-train-vae=scripts.train_vae:main",
            "aldm-train-diffusion=scripts.train_diffusion:main",
            "aldm-generate=scripts.generate:main",
            "aldm-evaluate=scripts.evaluate:main",
        ],
    },
)
