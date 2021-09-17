"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="shfl",
      version="0.1.0",
      description="Sherpa.ai Federated Learning Framework is an open-source framework for Machine Learning that is dedicated to data privacy "
                  "protection",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/sherpaai/Sherpa.ai-Federated-Learning-Framework",
      packages=find_packages(),
      install_requires=['numpy', 'emnist', 'scikit-learn>=0.23', 'pytest', 'pytest-cov', 'tensorflow>=2.2.0', 'scipy', 'six', 'pathlib2', 'torch>=1.7',
                        'pandas', 'multipledispatch'],
      python_requires='>=3.7')
