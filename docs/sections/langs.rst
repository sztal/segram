Supported languages and models
==============================

Since `segram` is based on `spacy`_ as its engine for solving core NLP
tasks, in order to do any work one needs to download and install appropriate
language models.

English
-------

Currently **only English** is supported and the recommended models are:

`en_core_web_trf`

    **Main English model based on the transformer architecture.**
    It should be used as the main model for best results.

`en_core_web_lg`

    **Word vector model.** The transformer model is powerful, but it does
    not provide static word vectors, but only context-dependent vectors.
    Several methods implemented in `segram` require context-free word vectors,
    so they must be obatined from a different model.

`en_coreference_web_trf`

    **Coreference resolution model.** This is a separate model trained
    for solving the coreference resolution task. Importantly, it has to
    be in a version consistent with the requirements specified in the
    table **Core requirements (coreference resolution)**.


Once `spacy`_ is installed, the three language models can be downloaded
and installed quite easily:

.. code-block::

    # Core model based on the trasnformer architecture
    python -m download en_core_web_trf
    # Mode for word vectors (can be skipped if vector similarity methods are not needed)
    python -m download en_core_web_lg  # skip if word vectors are not needed
    # Coreference resolution model
    pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl


.. include:: /sections/links.rst
