"""Python setup.py for grid_neural_abstractions package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("grid_neural_abstractions", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="grid_neural_abstractions",
    version=read("grid_neural_abstractions", "VERSION"),
    description="Awesome grid_neural_abstractions created by Zinoex",
    url="https://github.com/Zinoex/grid-neural-abstractions/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Zinoex",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["grid_neural_abstractions = grid_neural_abstractions.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
