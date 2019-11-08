# Contributing

Thank you for taking the time to contribute to this project!

1. Install the development requirements. Make sure you do this in a virtual environment!

```pip install -r dev-requirements.txt```


2. Install TensorFlow or TensorFlow-gpu. 

```pip install tensorflow-gpu==1.7.1```

or

```pip install tensorflow==1.7.1```

Tensorflow versions `1.7.1`, `1.10`, and `1.13.1` pass the tests.


3. After making desired changes to the code, run the unit tests. In the project root directory, please execute the following

```pytest bubs/```

4. Lint

```pylint --rcfile=.pylintrc .```

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
Please report unacceptable behavior to [bubs@kensho.com](bubs@kensho.com).

## Contributor License Agreement

Each contributor is required to agree to our Contributor License Agreement, to ensure that their contribution may be safely merged into the project codebase and released under the existing code license. This agreement does not change contributors' rights to use the contributions for any other purpose -- it is simply used for the protection of both the contributors and the project.

## Style Guide

This project follows the [Google Python style guide](http://google.github.io/styleguide/pyguide.html).

All python files in the repository must display the copyright of the project, to protect the terms of the license. Please make sure that your files start with a line like:

```# Copyright 20xx-present Kensho Technologies, LLC.```
