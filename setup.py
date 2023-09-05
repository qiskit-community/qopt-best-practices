import os
import setuptools

long_description = """Repository for best practices in quantum optimization."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(
    os.path.dirname(__file__), "qopt_best_practices", "VERSION.txt"
)
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="qopt_best_practices",
    version=VERSION,
    description="Best practices quantum opt.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ElePT/q-optimization-best-practices",
    author="Quantum optimization working group",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit quantum optimization",
    packages=setuptools.find_packages(
        include=["qopt_best_practices", "qopt_best_practices.*"]
    ),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
