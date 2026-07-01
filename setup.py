from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="2d3d",
    version="0.3.0",
    description="Convert 2D videos to stereoscopic 3D using AI depth estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JulesMellot/2D3D",
    py_modules=["main", "pipeline", "depth", "gui_app"],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={"console_scripts": ["2d3d=main:main"]},
)
