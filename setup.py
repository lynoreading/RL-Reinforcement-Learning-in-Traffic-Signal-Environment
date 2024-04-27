import os

import setuptools

from src import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "track_env": [
        "gymnasium"
    ],
    "track_agent": [
        "minigrid",
        "gymnasium[box2d]"
    ],
    "dev": [
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl_sphinx_theme",
        # Others
        "mypy",
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        ]
}

setuptools.setup(
    name='TrafficEnv',
    author='Reward Chasers',
    author_email='',
    description='Env for RL exam project',
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    url='',
    project_urls='',
    version=1.0.0,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "tqdm",
        "rich",
        "panadas"
        "gym",
        "stable-baselines3[extra]"
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
