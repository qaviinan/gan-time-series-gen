# setup.py

from setuptools import setup, find_packages

setup(
    name="high_freq_crypto_gan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==1.15.0",
        "numpy",
        "pandas",
        "click",
    ],
    entry_points={
        'console_scripts': [
            'crypto-gan=cli.main:main',
        ],
    },
    author="Qavi Inan",
    author_email="qaviinan@gmail.com",
    description="CLI tool for generating high-frequency crypto time-series data using GAN.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qaviinan/gan-time-series-gen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
