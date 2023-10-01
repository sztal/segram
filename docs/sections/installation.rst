Installation
============

At the command line via `pip`:

.. code-block::

    pip install segram
    # with CUDA GPU support
    pip install segram[gpu]
    # with experimental coreference resolution module
    pip install segram[coref]
    # with both
    pip install segram[gpu,coref]



Development version
-------------------

The current development version can be installed directly from the `Github repo`_

.. code-block::

    pip install "segram @ git+ssh://git@github.com/sztal/segram.git"
    # with CUDA GPU support
    pip install "segram[gpu] @ git+ssh://git@github.com/sztal/segram.git"
    # with experimental coreference resolution module
    pip install "segram[coref] @ git+ssh://git@github.com/sztal/segram.git"
    # with both
    pip install "segram[gpu,coref] @ git+ssh://git@github.com/sztal/segram.git"



Requirements
------------

.. list-table:: Core requirements
    :widths: 50 50
    :header-rows: 1

    * - Package
      - Version
    * - `python`
      - `>=3.11`
    * - `spacy`
      - `>=3.4`

The required Python version will not change in the future releases
for the foreseeable future, so before the package becomes fully
mature the dependency on `python>=3.11` will not be too demanding
(although it may be bumped to `>=3.12` as the new release is expected
soon as of time of writing - 29.09.2023).

Coreference resolution
~~~~~~~~~~~~~~~~~~~~~~

`Segram` comes with a coreference resolution component based on an
experimental model provided by `spacy-experimental`_ package.
However, both at the level of `segram` and `spacy`_ this is currently
an experimental feature, which comes with a significant price tag attached.
Namely, the acceptable `spacy`_ version is significantly limited
(see the table below). However, as `spacy-experimental` gets integrated
in the `spacy` core in the future, these constraints will be relaxed.


.. list-table:: Core requirements (coreference resolution)
    :widths: 50 50
    :header-rows: 1

    * - Package
      - Version
    * - `spacy`
      - `>=3.4,<3.5`
    * - `spacy-experimental`
      - `0.6.3`
    * - `en_coreference_web_trf`
      - `3.4.0a2`


.. include:: /links.rst

