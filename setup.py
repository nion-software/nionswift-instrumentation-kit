# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os


setuptools.setup(
    name="nionswift-instrumentation",
    version="0.16.0",
    author="Nion Software",
    author_email="swift@nion.com",
    description="A Nion Swift library for STEM instrumentation (Camera, Scan, Video, Microscope).",
    long_description=open("README.rst").read(),
    url="https://github.com/nion-software/nionswift-instrumentation-kit",
    packages=["nion.instrumentation", "nion.instrumentation.test", "nionswift_plugin.nion_instrumentation_ui"],
    package_data={"nionswift_plugin.nion_instrumentation_ui": ["resources/*", "manifest.json"]},
    install_requires=["nionswift>=0.14.0"],
    license='GPLv3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
    python_requires='~=3.6',
)
