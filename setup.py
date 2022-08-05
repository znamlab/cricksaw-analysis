from setuptools import find_packages, setup

setup(
    name="cricksaw_analysis",
    packages=find_packages(),
    install_requires=[
        'SimpleITK-Elastix>=2.0.0rc2',
        'numpy>=1.19.2',
        'pyyaml>=6.0',
        'pynrrd>=0.4.2',
        'matplotlib>=3.5.1',
    ],
)
