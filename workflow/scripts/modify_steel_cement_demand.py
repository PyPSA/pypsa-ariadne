

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
    year = int(snakemake.input.industrial_production.split("_")[-1].split(".")[0])
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

    mapping = {
        "gas boiler" : "Gas Boiler",
        "oil boiler" : "Oil Boiler",
        "air heat pump" : "Heat Pump|Electrical|Air",
        "ground heat pump" : "Heat Pump|Electrical|Ground",
        "biomass boiler" : "Biomass Boiler",
    }

    year = "2020"
    for tech in mapping:
        stock = ariadne.at[   
            f"Stock|Space Heating|{mapping[tech]}",
            year,
        ]

        peak = (
            stock * existing_heating.loc["Germany"].sum()
            / ariadne.at[f"Stock|Space Heating", year]

        )
        existing_heating.at["Germany", tech] = peak



    print(f"Heating demand after modification with {leitmodell}:", 
        existing_heating.loc["Germany"], sep="\n")


    existing_heating.to_csv(snakemake.output.existing_heating)
