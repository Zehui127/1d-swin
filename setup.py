from setuptools import setup, find_packages

setup(
    name="swin1d",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "torch",
        "einops",
        "transformers",
    ],
    author="Zehui Li",
    author_email="zehui.li22@imperial.ac.uk",
    description="A python package for 1D-Swin",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zehui127/1d-swin/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
# delete conda env
# conda env remove -n swin1d
