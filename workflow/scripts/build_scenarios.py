# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script reads in data from the IIASA database to create the scenario.yaml file


import ruamel.yaml
from pathlib import Path
import pandas as pd
import os

def get_shares(df, planning_horizons):
    # Get share of vehicles for transport sector - meglecting heavy duty vehicles
    # At the 
    total_transport = df.loc["DEMO v1", "Stock|Transportation|LDV"]
    tech_transport = df.loc["DEMO v1"].loc[[ 
        "Stock|Transportation|LDV|ICE",
        "Stock|Transportation|LDV|BEV",
        "Stock|Transportation|LDV|PHEV",
    ]]

    transport_share = tech_transport / total_transport
    transport_share = transport_share[planning_horizons]
    transport_share.set_index(pd.Index(["ICE", "BEV", "PHEV"]), inplace=True)

    # Get share of Navigation fuels
    total_navigation = \
        df.loc["REMIND-EU v1.1", "Final Energy|Bunkers|Navigation"] + \
        df.loc["DEMO v1", "Final Energy|Transportation|Domestic Navigation"]
    navigation_liquid = \
        df.loc["REMIND-EU v1.1", "Final Energy|Bunkers|Navigation|Liquids"] + \
        df.loc["DEMO v1", "Final Energy|Transportation|Domestic Navigation|Liquids"]
    
    navigation_h2 = df.loc["DEMO v1", "Final Energy|Transportation|Domestic Navigation|Hydrogen"]    

    h2_share = navigation_h2 / total_navigation
    liquid_share = navigation_liquid / total_navigation
    methanol_share = (1 - h2_share - liquid_share).round(6)
    
    naval_share = pd.concat(
            [liquid_share, h2_share, methanol_share]).set_index(
            pd.Index(["Oil", "H2", "MeOH"])
        )[planning_horizons]

    return transport_share, naval_share


def write_to_scenario_yaml(output, scenarios, transport_share, naval_share):
    # read in yaml file
    yaml = ruamel.yaml.YAML()
    file_path = Path(output)
    config = yaml.load(file_path)
    
    mapping_transport = {
        'PHEV': 'land_transport_fuel_cell_share',
        'BEV': 'land_transport_electric_share',
        'ICE': 'land_transport_ice_share'
    }
    mapping_navigation = {
        'H2': 'shipping_hydrogen_share',
        'MeOH': 'shipping_methanol_share',
        'Oil': 'shipping_oil_share',
    }

    for scenario in scenarios:
        for key in mapping_transport.keys():
            for year in transport_share.columns:
                config[scenario]["sector"][mapping_transport[key]][year] = round(transport_share.loc[key, year].item(), 4)
        for key in mapping_navigation.keys():
            for year in naval_share.columns:
                config[scenario]["sector"][mapping_navigation[key]][year] = round(naval_share.loc[key, year].item(), 4)

    # write back to yaml file
    yaml.dump(config, file_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_scenarios")

    # Set USERNAME and PASSWORD for the Ariadne DB
    ariadne = pd.read_csv(
        snakemake.input.ariadne_database,
        index_col=["model", "scenario", "region", "variable", "unit"]
    )
    ariadne.columns = ariadne.columns.astype(int)

    df = ariadne.loc[
        :, 
        snakemake.params.iiasa_scenario, 
        "Deutschland"]


    planning_horizons = [2020, 2025, 2030, 2035, 2040, 2045]
    transport_share, naval_share = get_shares(df, planning_horizons)

    scenarios = snakemake.params.scenario_name

    if "snakemake" in globals():
        filename = snakemake.input.scenario_yaml
    else:
        filename = "../config/scenarios.yaml"

    write_to_scenario_yaml(filename, scenarios, transport_share, naval_share)