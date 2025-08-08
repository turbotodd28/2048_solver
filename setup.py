from setuptools import setup, find_packages

setup(
    name="2048_solver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "stable-baselines3",
        "gymnasium",
        "numpy",
    ],
) 