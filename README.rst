=============================
Segram
=============================

Agent-oriented semantic grammar


Requirements
------------

* `Python3.10+`

It is recommended to set up a separate Conda environment
for working with `segram` in which the newest version of Python can be
easily installed.
See `this <https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html>`_
for more details.

Preparing environment
---------------------

Below are detailed instructions for setting up a Conda environment with
working `segram` installation and the simple English language model provided
by `spacy`. This is not the only proper way to prepare an environment but
only an example.

The instructions below assume that the `requirements.txt` file from this
repository is downloaded locally (you can simply copy-and-paste its content).

.. code-block:: bash

    # CREATE A CONDA ENV
    # WE USE NAME `segram-test` BUT YOU CAN USE ANY OTHER NAME
    conda create --name segram-test python=3.10

    # ACTIVATE THE ENVIRONMENT
    conda activate segram-test

    # DOWNLOAD DEPENDENCIES VIA PIP
    pip install -r requirements.txt

Now you can proceed to installing `segram` (below).

Installation
------------

.. code-block:: bash

    # OVER HTTPS (DEPRECATED)
    pip install git+https://github.com/sztal/segram.git
    # OVER SSH (RECOMMENDED)
    pip install git+ssh://git@github.com/sztal/segram.git
