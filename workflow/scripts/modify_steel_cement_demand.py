

import pandas as pd

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("modify_steel_cement_demand",
                                   planning_horizons=2020)

    # leitmodell for steel and cement demand
    leitmodell="FORECAST v1.0"
    year = snakemake.input.industrial_production.split("_")[-1].split(".")[0]
    existing_industry = pd.read_csv(snakemake.input.industrial_production, index_col=0)

    ariadne = pd.read_csv(
        snakemake.input.ariadne,
        index_col=["model", "scenario", "region", "variable", "unit"]
    ).loc[
        leitmodell,
        snakemake.config["iiasa_database"]["reference_scenario"],
        "Deutschland",
        :,
        "Mt/yr",
    ]

    print(
        "German demand before modification", 
        existing_industry.loc["DE", ["Cement", "Electric arc", "Integrated steelworks", "DRI + Electric arc"]], sep="\n")
    # get cement and write it into the existing industry dataframe
    existing_industry.loc["DE", "Cement"] = ariadne.loc["Cement", year]
    
    # get steel ratios from existing_industry
    steel = existing_industry.loc["DE", ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]]
    ratio = steel/steel.sum()

    # multiply with steel production including primary and secondary steel since distinguishing is taken care of later
    existing_industry.loc["DE", ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]] = ratio * ariadne.loc["Production|Steel", year]

    print(
        "German demand after modification", 
        existing_industry.loc["DE", ["Cement", "Electric arc", "Integrated steelworks", "DRI + Electric arc"]], sep="\n")

    existing_industry.to_csv(snakemake.output.industrial_production)
