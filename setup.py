from setuptools import setup, find_namespace_packages

setup(
    name="neural",
    version="0.1.0",
    description="A set of pretrained deep learning models for vision tasks.",
    license="BSD3",
    author="Bernardo Lourenço",
    author_email="bernardo.lourenco@ua.pt",
    python_requires=">=3.6.0",
    url="https://github.com/bernardomig/neural",
    packages=find_namespace_packages(
        exclude=["tests", ".tests", "tests_*", "scripts"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
