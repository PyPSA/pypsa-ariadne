
import pandas as pd
import re
import os
import logging

if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "modify_cost_data",
            planning_horizons="2030",
            file_path="../data/costs/",
            file_name="costs_2030.csv",
            cost_horizon="mean",
            )
    logger = logging.getLogger(__name__)

    # read in cost data from technology-data library
    filepath = snakemake.params.file_path + snakemake.params.cost_horizon
    costs = os.path.join(filepath, snakemake.params.file_name)
    costs = pd.read_csv(costs, index_col=[0, 1]).sort_index()

    # Pessimist: costs are taken from + 5 years
    if snakemake.params.cost_horizon == "pessimist":
        match = re.search(r"costs_(\d{4})-modifications\.csv", snakemake.input[0])
        if int(match.group(1)) > 2020:
            new_year = int(match.group(1)) + 5
            new_filename = re.sub(r'costs_\d{4}-modifications\.csv', f"costs_{new_year}-modifications.csv", snakemake.input[0])
            modifications = pd.read_csv(new_filename, index_col=[0, 1]).sort_index()
            costs.loc[modifications.index] = modifications
            logger.warning(f"Pessimistic cost scenario for {match.group(1)}.")
        else:
            modifications = pd.read_csv(snakemake.input.modifications, index_col=[0, 1]).sort_index()
            costs.loc[modifications.index] = modifications

    # Optimist: costs are taken from - 5 years
    elif snakemake.params.cost_horizon == "optimist":
        match = re.search(r"costs_(\d{4})-modifications\.csv", snakemake.input[0])
        if int(match.group(1)) > 2020:
            new_year = int(match.group(1)) - 5
            new_filename = re.sub(r'costs_\d{4}-modifications\.csv', f"costs_{new_year}-modifications.csv", snakemake.input[0])
            modifications = pd.read_csv(new_filename, index_col=[0, 1]).sort_index()
            costs.loc[modifications.index] = modifications
            logger.warning(f"Optimistic cost scenario for {match.group(1)}.")
        else:
            modifications = pd.read_csv(snakemake.input.modifications, index_col=[0, 1]).sort_index()
            costs.loc[modifications.index] = modifications

    # Mean: costs are taken from the same year
    elif snakemake.params.cost_horizon == "mean":
        modifications = pd.read_csv(snakemake.input.modifications, index_col=[0, 1]).sort_index()
        costs.loc[modifications.index] = modifications
        logger.warning("Mean cost scenario.")

    print(modifications)
    print( costs.loc[modifications.index])

    costs.to_csv(snakemake.output[0])
