from setuptools import setup, find_packages

setup(
    name="altwer",
    version="0.3.0",
    description="A package to calculate WER with multiple reference options.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Per Kummervold",
    author_email="Per.Kummervold@nb.no",
    url="https://github.com/peregilk/altwer",
    packages=find_packages(),
    install_requires=[
        "jiwer>=2.0.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
