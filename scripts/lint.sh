#!/usr/bin/env bash
set -eu
pylint --rcfile=.pylintrc bubs
pydocstyle --config=.pydocstyle_test bubs
pydocstyle --config=.pydocstyle bubs
isort --recursive --diff bubs
flake8 --config=setup.cfg bubs
