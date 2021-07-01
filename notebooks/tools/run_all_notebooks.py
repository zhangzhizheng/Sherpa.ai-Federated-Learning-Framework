#!/usr/bin/env python
"""
Runs and validates all the notebooks.
LICENSE: Public Domain

Usage:
1. You need to change  kernel_name=<virtual_kernel_name_you_are_running_on>
2. From terminal, run the script as
    ./run_all_notebooks.py [-s <start_number> -e <end_number>]
"""


import os
import glob
import time
import sys
import getopt
import traceback

import nbconvert
import nbformat

ep = nbconvert.preprocessors.ExecutePreprocessor(
    extra_arguments=["--log-level=40"],
    timeout=-1,
    kernel_name="SherpaFL_py37"
)


def run_notebook(path):
    path = os.path.abspath(path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
    except Exception as e:
        print("\nException raised while running '{}'\n".format(path))
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


def main(argv):
    notebook_paths = glob.glob('../**/*.ipynb', recursive=True)
    start_number = 0
    end_number = len(notebook_paths)
    try:
        opts, args = getopt.getopt(argv, "hs:e:")
    except getopt.GetoptError:
        print('Usage: ./run_all_notebooks.py [-s <start_number> -e <end_number>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: ./run_all_notebooks.py [-s <start_number> -e <end_number>]')
            sys.exit()
        elif opt in "-s":
            start_number = int(arg)
        elif opt in "-e":
            end_number = int(arg)

    print('\n')
    print('The following notebooks are set to run:\n')
    for i in range(start_number, end_number):
        print('{}. '.format(i) + notebook_paths[i])
    print('\n')
    print('Warning: Some notebook might take up to 30 minutes to finish.\n')

    for i in range(start_number, end_number):
        root, ext = os.path.splitext(os.path.basename(notebook_paths[i]))
        if root.endswith('_'):
            continue
        s = time.time()
        sys.stdout.write('Now running notebook {} : '
                         .format(i) +
                         notebook_paths[i] + '\n')
        sys.stdout.flush()
        run_notebook(notebook_paths[i])
        sys.stdout.write(' -- Finished in {}s.\n'.format(int(time.time()-s)))
        print('\n')

    print('\n\033[92m'
          '==========================='
          ' Notebook testing done. '
          '==========================='
          '\033[0m')


if __name__ == "__main__":
    main(sys.argv[1:])
