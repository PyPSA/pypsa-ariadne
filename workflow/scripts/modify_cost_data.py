
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
            cost_horizon="pessimist",
            run="KN2045_Bal_v4"
            )
    logger = logging.getLogger(__name__)

    # read in cost data from technology-data library
    costs = os.path.join(
        snakemake.params.file_path,     
        snakemake.params.cost_horizon, 
        snakemake.params.file_name)

    # cost_horizon is a setting for technology-data and specifies either
    # mean, pessimist or optimist cost scenarios
    # the cost modifications file contains specific cost assumptions for
    # germany, developed in the ARIADNE project
    # here pessimist and optimistic scenarios correspond to a delay or a 
    # speed up in cost reductions

    costs = pd.read_csv(costs, index_col=[0, 1]).sort_index()

    matched_year = int(re.search(
            r"costs_(\d{4})-modifications\.csv", 
            snakemake.input.modifications
        ).group(1))

    if matched_year <= 2020 or snakemake.params.cost_horizon == "mean":
        logger.warning(f"Mean cost scenario for {matched_year}.")
        new_year = matched_year
    elif snakemake.params.cost_horizon == "pessimist":
        logger.warning(f"Pessimistic cost scenario for {matched_year}.")
        new_year = matched_year + 5
    elif snakemake.params.cost_horizon == "optimist":
        logger.warning(f"Optimistic cost scenario for {matched_year}.")
        new_year = matched_year - 5
    else:
        raise ValueError("Invalid specification of cost options.")

    new_filename = re.sub(
        r'costs_\d{4}-modifications\.csv', 
        f"costs_{new_year}-modifications.csv", 
        snakemake.input.modifications)
    modifications = pd.read_csv(new_filename, index_col=[0, 1]).sort_index()
    if snakemake.params.NEP == 2021:
        modifications = modifications.query("source != 'NEP2023")
    elif snakemake.params.NEP == 2023:
        modifications = modifications.query("source != 'NEP2021")
    else:
        logger.warning(f"NEP year {snakemake.params.NEP} is not in modifications file. Falling back to NEP2021.")
        modifications = modifications.query("source != 'NEP2023")

    costs.loc[modifications.index] = modifications

    print(costs.loc[modifications.index])

    costs.to_csv(snakemake.output[0])
