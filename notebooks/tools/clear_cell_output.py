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

#!/usr/bin/env python
"""strip outputs from an IPython Notebook
Opens a notebook, strips its output, and writes the outputless version to the original file.
Useful mainly as a git filter or pre-commit hook for users who don't want to track output in VCS.
This does mostly the same thing as the `Clear All Output` command in the notebook UI.
LICENSE: Public Domain
"""

import io
import sys

try:
    # Jupyter >= 4
    from nbformat import read, write, NO_CONVERT
except ImportError:
    # IPython 3
    try:
        from IPython.nbformat import read, write, NO_CONVERT
    except ImportError:
        # IPython < 3
        from IPython.nbformat import current

        def read(f, as_version):
            return current.read(f, 'json')

        def write(nb, f):
            return current.write(nb, f, 'json')


def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def strip_output(nb):
    """strip the outputs from a notebook object"""
    nb.metadata.pop('signature', None)
    for cell in _cells(nb):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    return nb


def replace_relative_links(nb):
    """make all relative links to files, global by replacing them with github ones"""
    for cell in _cells(nb):
        if cell['cell_type'] == 'markdown':
            cell['source'] = cell['source'].replace(
                "(../../", "(https://github.com/sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/")
    return nb


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        with io.open(filename, 'r', encoding='utf8') as f:
            nb = read(f, as_version=NO_CONVERT)
        nb = strip_output(nb)
        # nb = replace_relative_links(nb)
        with io.open(filename, 'w', encoding='utf8') as f:
            write(nb, f)
