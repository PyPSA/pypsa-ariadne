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

The snakemake rule `retrieve_ariadne_database` logs into the IIASA Database via the [`pyam`](https://pyam-iamc.readthedocs.io/en/stable/tutorials/iiasa.html) package. The credentials for logging into this database have to be stored locally on your machine with `ixmp4`. To do this, run

```
ixmp4 login <username>
```

You will be prompted to enter your `<password>`. 

Caveat: These credentials are stored on your machine in plain text.

## Run the analysis

Before running any scenarios, the rule `build_scenarios` must be executed. This will write the file `config/scenarios.automated.yaml` which includes transport shares and ksg goals from the iiasa database as well as the information from the file `config/scenarios.manual.yaml`.

    snakemake -call build_scenarios -f

Note that the hierarchy of scenario files is the following: `scenarios.automated.yaml` > `config.yaml` > `config.default.yaml`
Changes in the file `scenarios.manual.yaml` are only taken into account if the rule `build_scenarios` is executed.

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
