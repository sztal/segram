# Segram: _a framework for semantic grammar and narrative analysis_

> **Note**
> This project is still in an early stage of development,
> so one should expect significant changes in the future,
> including backward incompatible ones. That said, the general
> concepts and design principles should remain the same or be extended,
> not changed or limited. Thus, the package is suitable for experimental
> usage.

**Segram** is a software implementation of a framework for automated
semantics-oriented grammatical analysis of text data. It is implemented
in Python and based on the excellent [spacy](https://spacy.io/)
package, which is used to solve core NLP tasks such as tokenization,
lemmatization, dependency parsing and coreference resolution.

## Core requirements

| Package                  | Version            |
| ------------------------ | ------------------ |
| `python`                 | `>=3.11`           |
| `spacy`                  | `>=3.4,<=3.5`      |
| `spacy-experimental`     | `0.6.1`            |
| `en_coreference_web_trf` | `3.4.0a2`          |

The constraints on the versions of `spacy` and `spacy-experimental`
are imposed by the fact that currently `segram` depends on one specific
version of coreference resolution component offered by `spacy`,
which is `en_coreference_web_trf(3.4.0a2)` model. However, these
requirements will be relaxed as the experimental coref component
gets integrated into the core of `spacy`.

> **Warning**
> Due to a bug in `spacy-experimental=0.6.1` GPU acceleration is
> currently not supported, i.e. using `spacy.prefer_gpu()` will
> result in errors during coreference resolution.

This is a very ealy prototype of an implementation of a conceptual
framework for automated analysis of natural language text data
from the narrative perspective. The main focus is on describing and
representing text in terms of actors and actions they perform, both
as active and passive sides. The main aim is to have a systematic
framework and related toolset for narrative analysis that will serve
several interrelated goals:

1. faclitating in-depth reading of human analysts by:
2. providing easy-to-use, transparent tools for
   automated information extraction and text filtering aligned with
   intuitive notions human of semantics, including people not trained
   in theoretical linguistics and computer science.
3. generating interpretable descriptive statistics for summarizing
   large text corpora in terms of their narrative meaning, that is,
   actors and actions they perform.

Additionally, the data produced by `segram` could be, in principle,
used to derive rich graphical data structures in the form of multilayer
hypergraphs and/or power graphs. This is a research avenue that we will
be exploring actively in the future.

Architecture
============

`segram` follows a a gray-box model approach and was designed as a
combination of a typical NLP black box model (provided by an excellent
`spacy <https://spacy.io/>`_ package for Python, which is used to handle
tokenization, provide liinguistic annotations as well as perform basic
entity recognition and coreference resolution. Then, on top of that
`segram` applies a set of transparent deterministic rules to build
a rich data structures attempting to represent the basic narrative
element of texts.

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

Now you can proceed to installing `segram` (below).
Currently the package can be accessed only from its
`Github repo <https://github.com/sztal/segram>`_.
The below command installs the package with all core dependencies.
However, the language models for English need to be installed separately
(as these are relatively large files).

Installation
------------

.. code-block:: bash

    pip install git+ssh://git@github.com/sztal/segram.git

    # Install core language model for English
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.4.1/en_core_web_trf-3.4.1-py3-none-any.whl
    # Install language model for coreference resolution
    pip intsall https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl


Usage example
-------------

The package is still under very active development and its still
lacking several features that are necessary for making it fully functional.
Nonetheless, below we present a very simple exmaple of focused on extracting
information about actions performed by a specific agent.

.. code-block:: python

    import spacy
    from segram import Story

    spacy.prefer_gpu() # GPU acceleration is recommended but requires
                       # but everything runs also on CPU

    NLP = spacy.load("en_core_web_trf")
    NLP.add_pipe("segram")
    NLP.add_pipe("segram_coref")

    # SAMPLE TEXT
    text = (
        "The Blue Whales just played their first baseball game of the new season; "
        "I believe there is much to be excited about. "
        "Although they lost, it was against an excellent team that had won "
        "the championship last year. The Blue Whales fell behind early but "
        "showed excellent teamwork and came back to tie the game. "
        "The team had 15 hits and scored 8 runs. "
        "That’s excellent! Unfortunately, they had 5 fielding errors, "
        "which kept the other team in the lead the entire game. "
        "The game ended with the umpire making a bad call, "
        "and if the call had gone the other way, the Blue Whales might have "
        "actually won the game. It wasn’t a victory, but I say the Blue Whales "
        "look like they have a shot at the championship, especially if they "
        "continue to improve."
    )

Now we will extract all narrative action elements in which The Blue Whales
team is the subject (i.e. action-performing agent). This will also include
sentences in which the team appears only through a coreference.

.. code-block:: python

    doc = NLP(text)._.segram
    sents = [ s.grammar() for s in doc.sents ]
    story = Story.from_sents(doc, sents)

    for action in story.actions:
        for p in action.iter_relations():
            if p.subject and "Blue Whales" in p.subject.to_str():
                print(p)

Different colors are used to mark parts corresponding to different
semantic/syntactic roles such as subjects, verbs or direct objects
(or general descriptions such as adjectives and adverbs).

We can get a slightly better view of the underlying data by inspecting
the `data` attribute of the found action elements.

.. code-block:: python

    for action in story.actions:
        for p in action.iter_relations():
            if p.subject and "Blue Whales" in p.subject.to_str():
                print(p)
                print(p.data, end="\n\n")
