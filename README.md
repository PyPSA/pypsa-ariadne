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

## Provide login details

The snakemake rule `retrieve_ariadne_scenario_data` logs into the IIASA Database. It requires a USERNAME and a PASSWORD which should be set as environment variables in your local shell configuration. To do that on Linux open your `.bashrc` with a text editor, e.g., with

```
vim ~/.bashrc
```

and then add the following two lines to the end of that file:

```
export IIASA_USERNAME='USERNAME'
export IIASA_PASSWORD='PASSWORD'
```

Fill in the correct login details and don't forget the quotation marks. You might have to restart your terminal session / vscode window for the new variables to become available. 

**Caution for vscode users**: If you want to use the environment variables in an Interactive Python Session, another step might be required depending on your local config. Create a file `.env` in the working directory and add the lines:
```
IIASA_USERNAME='USERNAME'
IIASA_PASSWORD='PASSWORD'
```
Details on Python environment variables in VSCode can be found here: https://code.visualstudio.com/docs/python/environments#_environment-variables


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
