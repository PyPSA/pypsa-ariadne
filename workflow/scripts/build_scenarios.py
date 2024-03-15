# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script reads in data from the IIASA database to create the scenario.yaml file

import pyam
import yaml
import pandas as pd
import os

def get_shares(df):
    # Get share of vehicles for transport sector - meglecting heavy duty vehicles
    total_transport = df.loc["Stock|Transportation|LDV"]
    tech_transport = df.loc[["Stock|Transportation|LDV|ICE",
                "Stock|Transportation|LDV|BEV",
                "Stock|Transportation|LDV|PHEV",]]

    transport_share = tech_transport / total_transport
    transport_share = transport_share[[2020, 2025, 2030, 2035, 2040, 2045]]
    transport_share.reset_index(drop=True, inplace=True)
    transport_share.set_index(pd.Index(["ICE", "BEV", "PHEV"]), inplace=True)

    # Get share of Navigation fuels
    total_navigation = df.loc[["Final Energy|Bunkers|Navigation",
                         "Final Energy|Transportation|Domestic Navigation"]].sum(axis=0)
    navigation_liquid = df.loc[["Final Energy|Bunkers|Navigation",
                         "Final Energy|Transportation|Domestic Navigation|Liquids"]].sum()
    navigation_h2 = df.loc[["Final Energy|Transportation|Domestic Navigation|Hydrogen"]]    

    h2_share = navigation_h2 / total_navigation
    liquid_share = pd.DataFrame(navigation_liquid / total_navigation, columns=['Oil']).transpose()

    h2_share.reset_index(drop=True, inplace=True)
    h2_share.set_index(pd.Index(["H2"]), inplace=True)
    naval_share = pd.concat([liquid_share, h2_share])
    naval_share.loc["MeOH"] = 0.0
    naval_share = naval_share[[2020, 2025, 2030, 2035, 2040, 2045]]

    return transport_share, naval_share


def write_to_scenario_yaml(output, scenarios, transport_share, naval_share):
    # read in yaml file
    with open("config/scenarios.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
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
    with open(output, 'w') as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_scenarios")

    # Set USERNAME and PASSWORD for the Ariadne DB
    pyam.iiasa.set_config(
        snakemake.params.iiasa_usr, 
        snakemake.params.iiasa_pwd,
    )

    model_df= pyam.read_iiasa(
        "ariadne_intern",
        model=snakemake.params.iiasa_model,
        scenario=snakemake.params.iiasa_scenario,
    ).timeseries()

    df = model_df.loc[snakemake.params.iiasa_model, snakemake.params.iiasa_scenario, "Deutschland"]

    transport_share, naval_share = get_shares(df)

    scenarios = snakemake.params.scenario_name
    output = snakemake.input[0]
    write_to_scenario_yaml(output, scenarios, transport_share, naval_share)