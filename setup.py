# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os

setuptools.setup(
    name="nionswift-instrumentation-kit",
    version="0.1.0",
    author="Nion Software",
    author_email="swift@nion.com",
    description="Library and UI kit for STEM instrumentation (Camera, Scan, Microscope) with Nion Swift.",
    url="https://github.com/nion-software/nionswift-instrumentation-kit",
    packages=["nion.instrumentation", "nion.instrumentation.test", "nionswift_plugin.nion_instrumentation_ui"],
    package_data={"nionswift_plugin.nion_instrumentation_ui": ["resources/*"]},
    install_requires=[],
    license='GPLv3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
    ],
    include_package_data=True,
    test_suite="nion.instrumentation.test",
    python_requires='~=3.5',
)
