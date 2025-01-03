from setuptools import setup, find_packages

setup(
    name="altwer",
    version="0.1.0",
    description="A package to calculate WER with multiple reference options.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/altwer",
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
