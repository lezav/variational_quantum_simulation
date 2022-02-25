#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as readme:
    description = readme.read()

setuptools.setup(
    name='varqus',
    version='0.0.1',
    author='VarQus Team: QuantumLegion',
    author_email='lezhav@gmail.com',
    description='A package for Variational Quantum Simulation',
    long_description=description,
    long_description_content_type="text/markdown",
    url='https://github.com/lezav/variational_quantum_simulation',
    license='Apache Licence 2.0',
    packages=['varqus'],
    install_requires=['qiskit', 'numpy', 'scipy'],
    include_package_data=True,
)
