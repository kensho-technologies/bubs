# Copyright 2019-present Kensho Technologies, LLC.
import codecs
import logging
import os
import re

from setuptools import find_packages, setup


#  https://packaging.python.org/guides/single-sourcing-package-version/
#  #single-sourcing-the-version
PACKAGE_NAME = "bubs"


logger = logging.getLogger(__name__)


def read_file(filename):
    """Read package file as text to get name and version"""
    # intentionally *not* adding an encoding option to open
    # see here:
    # https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, PACKAGE_NAME, filename), "r") as f:
        return f.read()


def find_version():
    """Only define version in one place"""
    version_file = read_file("__init__.py")
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_long_description():
    """Return the content of the README.rst file."""
    return read_file('../README.md')


REQUIRED_PACKAGES = ["funcy>=1.10", "numpy>=1.10.0", "segtok>=1.5.7", "tensorflow>=1.13"]

setup(
    name=PACKAGE_NAME,
    version=find_version(),
    description='Keras Implementation of Flair\'s Contextualized Embeddings',
    long_description=find_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/kensho-technologies/bubs',
    author='Kensho Technologies, LLC.', author_email='bubsr@kensho.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    package_data={"": ["tests/dummy_weights.npz"]},
    dependency_links=[],
)
