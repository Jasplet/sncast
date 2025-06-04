from setuptools import setup, find_packages

setup(
    name="sncast",
    version="1.1.0",
    description="Software for modelling earthquake detection capability for a seismic network",
    author="Martin Molhoff, Joseph Asplet",
    author_email="joseph.asplet@earth.ox.ac.uk",
    insitution="University of Oxford",
    url="https://github.com/Jasplet/sncast",
    # URL to the project's repository (if available)

    # Automatically find all packages (folders with __init__.py)
    # in your project
    packages=find_packages(include=['sncast.*']),

    # Include additional files listed in MANIFEST.in
    include_package_data=True,

    # Project dependencies (install these when the package is installed)
    install_requires=[
        "numpy>=1.26.4",          # Example of a required package
        "obspy>=1.4.1",    # Specify version ranges, e.g., pandas 1.0 or higher
    ],

    # Classifiers for metadata, useful for PyPI (optional, but recommended)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL v3",
        "Operating System :: OS Independent",
    ],

    # Minimum Python version requirement (optional)
    python_requires='>=3.11',

    license="GNU General Public License v3.0 (GPLv3)",
)
