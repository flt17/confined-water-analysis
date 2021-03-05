from setuptools import setup, find_packages

setup(
    name="confined-water-analysis",
    version="0.1.0",
    url="https://tbd.com",
    author="Fabian Thiemann",
    author_email="flt17@imperial.ac.uk",
    description="Analysis code to evaluate the structural and dynamical behaviour of confined water",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "MDAnalysis", "ase", "findpeaks"],
)
