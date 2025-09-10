from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="2d3d",
    version="0.2.0",
    author="Open Source Community",
    author_email="opensource@example.com",
    description="An open-source tool for converting 2D videos to stereoscopic 3D while preserving quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/2d3d",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "2d3d=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["README.md", "requirements.txt"],
    },
)