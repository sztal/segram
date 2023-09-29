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

## Main use cases and features

* Automated grammatical analysis in terms of phrases/clauses focused
  on detecting actions as well as subjects and objects of those actions.
* Flexible filtering and matching with queries expressible
  in terms of properties of subjects, verbs, objects, prepositions and
  descriptions applicable at the levels of individual phrases and entire
  sentences.
* Semantic-oriented organization of analyses in terms of stories
  and frames.
* Data serialization framework allowing for reconstructing all `segram`
  data after an initial parsing without access to any `spacy` language
  model.
* Structured vector similarity model based on weighted averages of
  cosine similarities between different components of phrases/sentences
  (several algorithms based on somewhat different notions of what
  it means for sentences or phrases to be similar are available).
* Structured vector similarity model for comparing documents in terms
  of sequentially shifting semantics.
* Hypergraphical representation of grammatical structure of sentences.



## Core requirements

| Package                  | Version            |
| ------------------------ | ------------------ |
| `python`                 | `>=3.11`           |
| `spacy`                  | `>=3.4`            |

The required Python version will not change in the future releases
for the foreseeable future, so before the package becomes fully
mature the dependency on `python>=3.11` will not be too demanding
(although it may be bumped to `>=3.12` as the new release is expected
soon as of time of writing - 29.09.2023).

### Core requirements (coreference resolution)

`Segram` comes with a coreference resolution component based on an
experimental model provided by `spacy-experimental` package.
However, both at the level of `segram` and `spacy` this is currently
an experimental feature, which comes with a significant price tag attached.
Namely, the acceptable `spacy` version is significantly limited
(see the table below). However, as `spacy-experimental` gets integrated
in the `spacy` core in the future, these constraints will be relaxed.


| Package                  | Version            |
| ------------------------ | ------------------ |
| `spacy`                  | `>=3.4,<3,5`       |
| `spacy-experimental`     | `0.6.3`            |
| `en_coreference_web_trf` | `3.4.0a2`          |


## Supported models and languages

Currently, only English is supported and `segram` was tested on models:

* `en_core_web_trf>=3.4.1` (transformer-based model for the general NLP)
* `en_core_web_lgl>=3.4.1` (used for context-free word vectors)
* `en_coreference_web_trf==3.4.0a2` (for coreference resolution)


## Installation


### PyPI

```bash
pip install segram
python -m download en_core_web_trf
python -m download en_core_web_lg  # skip if word vectors are not needed
```

#### With coreference resolution

```bash
pip install segram[coref]
python -m download en_core_web_trf
python -m download en_core_web_lg  # skip if word vectors are not needed
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.4.1/en_core_web_lg-3.4.1-py3-none-any.whl
```

#### With GPU acceleration

```bash
pip install segram[gpu]
# OR in the case when coreference resolution is required
pip install segram[coref,gpu]
```

The rest of commands should remain the same depending on whether
coreference installation or not is used.

### Github (development version)

```bash
pip install git+ssh://git@github.com/sztal/segram.git
```

#### With coreference resolution

## Basic usage

```python
import spacy
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("segram", config={
    "vectors": "en_core_web_lg"
})
nlp.add_pipe("segram_coref")

# Get standard 'spacy' document
doc = nlp(
    "The merchants travelled a long way to buy spices "
    "and rest in our taverns."
)
# Convert it to segram 'grammar' document
doc = doc._.segram
doc
```

The code above parses the text using `spacy` and additionally applies
further processing pipeline components defined by `segram`. They inject
many additional functionalities into standard `spacy` tokens.
In particular, `Doc` instances are enhanced with a special extension
property `._.segram`, which converts them to `segram` grammar documents.
Note that the printing results is different - the output is colored!

The colors denote the partition of the document into **components**,
which are groups of related tokens such as each non-leaf token in the
dependency tree of a sentence is the head of one component which also
controls all its leaf children. The following (default) color scheme
is used:

* <span style="color:orange">**Noun components**</span> `#FF0000`
* <span style="color:red">**Verb components**</span> `#FFA500`
* <span style="color:violet">**Description components**</span> `#EE82EE`
* <span style="color:limegreen">**Preposition components**</span> `#32CD32`

Components are further organized into phrases, which are higher-order
and more semantically-oriented units. Crucially, while components are
non-overlapping and form a partition of the sentence, the phrases
can be nested in each other and form a directed acyclic graph (DAG).

## Development

### Setting up environment

The [Github repository](https://github.com/sztal/segram)
provides [conda](https://docs.conda.io/en/latest/) environment files
for setting up development/testing environment. This installs also
testing dependencies such as [pytest](https://docs.pytest.org/en/7.4.x/),
which is used for running unit tests.

```bash
git clone git@github.com:sztal/segram.git
cd segram
conda env create -f environment.yml # default env name is 'segram'
conda activate segram
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
pip install --editable .
```

### Setting up environment (with coref)

```bash
git clone git@github.com:sztal/segram.git
cd segram
conda env create -f environment-coref.yml # default env name is 'segram'
# In this case the versions of all language models are fixed
# so they are installed automatically with the rest of the dependencies
conda activate segram
pip install --editable .
```

### Testing

Currently, `segram` is only moderately tested with a unit coverage rate
of about 70%. This will be improved in the future releases.

```
pytest          # run unit tests
coverage run    # run unit tests and gather coverage statistics
coverage report # show coverage report
```

### Makefile

`Makefile` defines many commands useful during development
(see the content of the file). However, Windows users may need
to modify it a bit to make it work.

## Overview

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
