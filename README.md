# Kopernikus-Projekt Ariadne - Gesamtsystemmodell PyPSA-Eur

Dieses Repository enthält das Gesamtsystemmodell für das Kopernikus-Projekt Ariadne, basierend auf der Toolbox PyPSA und dem Datensatz PyPSA-Eur. Das Modell bildet Deutschland mit hoher geographischer Auflösung, mit voller Sektorenkopplung und mit Integration in das europäische Energiesystem ab.

This repository contains the entire scientific project, including data sources and code. The philosophy behind this repository is that no intermediary results are included, but all results are computed from raw data and code.

## Clone the repository - including necessary submodules!

To start you need to clone the [PyPSA-Ariadne repository](https://github.com/PyPSA/pypsa-ariadne/). Since the repository relies on Git Submodules to integrate the PyPSA-Eur dataset as a basis on which to expand, you need to include the `--recurse-submodules` flag in your `git clone` command:

    git clone --recurse-submodules git@github.com:PyPSA/pypsa-ariadne.git

Alternatively, after having cloned the repository without activating submodules, you can run the two following commands:

    git submodule update --init --recursive

This command first initializes your local configuration file, second fetches all the data from the project(s) declared as submodule(s) (in this case, PyPSA-Eur) as well as all potential nested submodules, and third checks out the appropriate PyPSA-Eur commit which is defined in the PyPSA-Ariadne repository.

You can fetch and merge any new commits from the remote of the submodules with the following command:

    git submodule update --remote

More information on Git Submodules can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Getting ready

You need [mamba](https://mamba.readthedocs.io/en/latest/) to run the analysis. Using mamba, you can create an environment from within you can run it:

    mamba env create -f environment.yaml

## Run the analysis

For the first run open config.yaml and set

    enable:
        retrieve: true # set to false once initial data is retrieved
        retrieve_cutout: true # set to false once initial data is retrieved

and then run from main repository

    snakemake -call

This will run all analysis steps to reproduce results.

To generate a PDF of the dependency graph of all steps `build/dag.pdf` run:

    snakemake -c1 --use-conda -f dag

## Repo structure

* `config`: configurations used in the study
* `cutouts`: very large weather data cutouts supplied by atlite library
* `data`: place for raw data
* `resources`: place for intermediate/processing data for the workflow
* `results`: will contain all results (does not exist initially)
* `workflow`: contains the Snakemake workflow, including the submodule PyPSA-Eur

## License

The code in this repo is MIT licensed, see `./LICENSE.md`.
