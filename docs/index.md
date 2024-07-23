![](_static/logo.png)

## Introduction

Welcome to the documentation of The Arctic Ocean's Carbon Cycle!

This project investigates continuous fields of surface ocean pCO2 in the Arctic Oceann. The data stem from the MPIM-SOM-FFN product (Landschützer, 2016), which uses a feed-forward neural network to fill gaps in the field of pCO2 observations.

```{tip}
- Give a short introduction on what the project is about.
- Limit yourself to just a few sentences.
- Mention any preknowledge the user must have for the project.
- You may also give some instructions on how to navigate through the documentation.
```

## Getting Started

### Installation

For getting started in the fastest way possible, there are *Make* targets provided.
To set up the project, simply run the following commands from the main directory:

First, run

```bash
make conda-env
# or alternatively
make install-requirements
```

to install the required packages either via `conda` or `pip`, followed by

```bash
make src-available
```

to make the project's routines (located in `src`) available for import.

### Test code

You can run

```bash
make tests
```

to run the tests via `pytest`.

### Make data available

You may need to make data available under `data`.
These can be either copied directly into the directory or, if the files are too big and/or already reside somewhere your machine can access it directly, create links in `data`, pointing to the actual location of the data.

You should now be set to proceed with executing the scripts and notebooks. 🚀

## Data

```{tip}
Describe the data you use in the project.
```

## Workflow

```{tip}
Describe the workflow that should be followed to reproduce the results of your project.
```


## Contact

For any questions or issues, please contact me via git@markusritschel.de or open an [issue](https://github.com/markusritschel/arctic-ocean-pco2/issues).
