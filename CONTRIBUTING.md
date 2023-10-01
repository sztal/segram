# Contributing

Contributions are welcome, and they are greatly appreciated! Every little
bit helps, and credit will always be given. If you want to contribute to
this project, please make sure to read the contribution guidelines below.

## Types of contributions

## Report Bugs

Report bugs at https://github.com/sztal/segram/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Write Documentation

`Segram` could always use more documentation, whether as part of the
official `Segram` docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/sztal/segram/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

## Pull Request Guidelines

1. [Fork](https://github.com/sztal/segram/fork) the `segram`
   repo on GitHub.

2. Clone your fork locally:

```bash
git clone git@github.com:your_name_here/segram.git
```

3. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

4. When you're done making changes, check that your changes pass style
   and unit tests.

5. Commit your changes and push your branch to GitHub::

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

6. Submit a pull request through the GitHub website.

### Setting up testing and development environment

Contributing new features or bug fixes requires setting up a proper
development environment. Below is an instruction for how to do that.

The [Github repository](https://github.com/sztal/segram)
provides [conda](https://docs.conda.io/en/latest/) environment files
for setting up development/testing environment. This installs also
testing dependencies such as [pytest](https://docs.pytest.org/en),
which is used for running unit tests.

```bash
git clone git@github.com:sztal/segram.git
cd segram
conda env create -f environment.yml # default env name is 'segram'
conda activate segram
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
pip install --editable .
# OR to allow for GPU acceleration:
pip install --editable .["gpu"]
```

#### Setting up environment with coref

```bash
git clone git@github.com:sztal/segram.git
cd segram
conda env create -f environment-coref.yml # default env name is 'segram'
# In this case the versions of all language models are fixed
# so they are installed automatically with the rest of the dependencies
conda activate segram
pip install --editable .
# OR to allow for GPU acceleration:
pip install --editable .["gpu"]
```

#### Testing

Currently, `segram` is only moderately tested with a unit coverage rate
of about 70%. This will be improved in the future releases.

```
pytest          # run unit tests
coverage run    # run unit tests and gather coverage statistics
coverage report # show coverage report
```

#### Makefile

`Makefile` defines several commands useful during development
(see the content of the file). However, Windows users may need
to modify it a bit to make it work.

Below is the list of commands most useful for development:

* `make clean`
  * Remove all build artifacts and cache files. Good to run this after
    changing `pyproject.toml` or other packaging configuration.
* `make gzip-jsons` and `make gunzip-jsons`
  * compress and decompress `.json` resource files contained distributed
    with the package source code (they store patterns used to customize
    lemmatizers etc. and data for building cases used in unit tests).
* `make list-deps`
  * Runs a regular expression over all `segram` source files and lists
    explicit dependencies (`import` statements) used in the code.


#### Building documentation

First, install extra dependencies:

```bash
pip install requirements/docs.txt
```

And then run a dedicated make command:

```bash
make docs
```
