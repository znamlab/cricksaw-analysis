from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cricksaw_analysis")
except PackageNotFoundError:
    # package is not installed
    pass
