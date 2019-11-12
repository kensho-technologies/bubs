#!/usr/bin/env bash
err=0
trap 'err=1' ERR
pylint --rcfile=.pylintrc bubs
pydocstyle --config=.pydocstyle_test bubs
pydocstyle --config=.pydocstyle bubs
isort --recursive --diff bubs
flake8 --config=setup.cfg bubs
test $err = 0 # Return non-zero if any command failed
