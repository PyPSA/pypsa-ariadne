
import pandas as pd
import os

if __name__ == "__main__":
    if "snakemake" not in globals():
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_cost_data",
            planning_horizon="2030",
            file_path="../data/costs/",
            file_name="costs_2030.csv",
            cost_horizon="mean",
            )

    filepath = snakemake.params.file_path + snakemake.params.cost_horizon
    costs = os.path.join(filepath, snakemake.params.file_name)

    costs = pd.read_csv(costs, index_col=[0, 1]).sort_index()

    if snakemake.params.cost_horizon == "pessimist":
        # modifications are taken from + 5 years
        print("pessimist")
    elif snakemake.params.cost_horizon == "optimist":
        # modifications are taken from - 5 years
        print("optimist")

    elif snakemake.params.cost_horizon == "mean":
        modifications = pd.read_csv(snakemake.input.modifications, index_col=[0, 1]).sort_index()
        costs.loc[modifications.index] = modifications
    
    print(modifications)
    print( costs.loc[modifications.index])

    costs.to_csv(snakemake.output[0])
