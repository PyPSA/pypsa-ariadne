import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_aladin_data():
    """
    Retrieve the German mobility demand from the Aladin model.
    Sum over the subsectors Bus, LDV, Rail, and Truck for the fuels electricity, hydrogen, and synthetic fuels.
    """
    # get aladin data
    db = pd.read_csv(
        snakemake.input.ariadne,
        index_col=["model", "scenario", "region", "variable", "unit"]
    ).loc[
        "Aladin v1",
        snakemake.params.reference_scenario,
        "Deutschland",
        :,
        :,]
    year = snakemake.wildcards.planning_horizons

    subsectors = ["Bus", "LDV", "Rail", "Truck"]
    fuels = ["Electricity", "Hydrogen", "Liquids"]

    transport_demand = pd.Series(0, index=fuels)

    for fuel in fuels:
        for subsector in subsectors:
            key = f"Final Energy|Transportation|{subsector}|{fuel}"
            transport_demand.loc[fuel] += db.loc[key, year].iloc[0]
    
    transport_demand = transport_demand.div(3.6e-6) # convert PJ to MWh
    transport_demand["number_of_cars"] = db.loc["Stock|Transportation|LDV|BEV", year].iloc[0]

    return transport_demand

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_mobility_demand",
            simpl="",
            clusters=22,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Bal_v4"
        )

    logger.info("Retrieving German mobility demand from Aladin model.")
    # get aladin data
    aladin = get_aladin_data()   

    # get German mobility weighting
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    # only get German data
    pop_layout = pop_layout[pop_layout.ct == "DE"].fraction

    mobility_demand = pd.DataFrame(pop_layout.values[:, None] * aladin.values, index=pop_layout.index, columns=aladin.index)

    mobility_demand.to_csv(snakemake.output.mobility_demand)
