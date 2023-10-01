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

<p align="center">
<img src="docs/assets/images/printing.png" alt="Simple example of document parsing using `segram` and of printing a phrasal graph to the console" width="50%">
</p>

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

<p align="center">
<img src="docs/assets/images/hypergraph.png" alt="Representation of sentence
as a hypergraph of phrases" width="50%">
</p>



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

Currently, **only English is supported** and `segram` was tested on models:

* `en_core_web_trf>=3.4.1` (transformer-based model for the general NLP)
* `en_core_web_lgl>=3.4.1` (used for context-free word vectors)
* `en_coreference_web_trf==3.4.0a2` (for coreference resolution)


## Installation


### PyPI

```bash
pip install segram
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg  # skip if word vectors are not needed
```

#### With GPU support coreference resolution

```bash
pip install segram[coref,gpu]
# Just one of the two options can also be selected

# And language models
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
# The last one is a special model for the coref component
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl
```

### Github (development version)

```bash
pip install git+ssh://git@github.com/sztal/segram.git
# + downloading language models
```

#### With GPU and coreference resolution

```bash
pip install "segram[gpu,coref] @ git+ssh://git@github.com/sztal/segram.git"
```

### Dependencies for running example notebooks

```bash
pip install -r requirements/examples.txt
```


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
Note that the printing results is different now - the output is colored!

The colors denote the partition of the document into **components**,
which are groups of related tokens headed by a syntactically and/or
semantically important token. They are divided into four distinct types
which are marked with different colors when printing to the console.
The following (default) color scheme is:

* $\text{\color{orange}\bf Noun components}$
* $\text{\color{red}\bf Verb components}$
* $\text{\color{violet}\bf Description components}$
* $\text{\color{limegreen}\bf Preposition components}$

Components are further organized into phrases, which are higher-order
and more semantically-oriented units. Crucially, while components are
non-overlapping and form a partition of the sentence, the phrases
can be nested in each other and form a directed acyclic graph (DAG).

### Running examples

[Examples](examples/) are [Jupyter notebooks](https://jupyter.org/)
with some sample analyses and tutorials. Below are instructions for
setting up an environment sufficient for running the notebooks.

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

# Finally, install some extra dependencies used in the notebooks
pip install -r requirements/docs.txt
```

### Development and contributing

See [development and contributing guidelines](CONTRIBUTING.md).

## Feedback

If you have any suggestions or questions about `segram` feel free to email
me at `<stalaga@protonmail.com>`.

If you encounter any errors or problems, please also let me know!
[Open an Issue](https://github.com/sztal/segram/issues)
in the [GitHub repository](https://github.com/sztal/segram).


Authors
=======

* Szymon Talaga, <stalaga@protonmail.com>
