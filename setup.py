from setuptools import find_packages, setup

setup(
    name="cricksaw_analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "pynrrd",
        "matplotlib",
        "setuptools",
        "napari",
        "scikit-image",
        "tifffile",
        "brainglobe_napari_io",
        "cellfinder",
        "czifile",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
    ],
)
