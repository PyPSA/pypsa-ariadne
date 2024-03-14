# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script reads in data from the IIASA database to create the scenario.yaml file

import pyam
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


def write_to_scenario_yaml(output, scenario, transport_share, naval_share):
    # write the data to the output file
    with open(output, 'r') as f:
        lines = f.readlines()
    
    #get first occurence of scenario
    first = lines.index(scenario + ":\n")
    
    mapping_transport = {
        'PHEV': 'land_transport_fuel_cell_share',
        'BEV': 'land_transport_electric_share',
        'ICE': 'land_transport_ice_share'
    }
    
    # transport sector
    start_index = []
    for i, line in enumerate(lines):
        if line.strip() == "land_transport_fuel_cell_share:":
            start_index.append(i)
        elif line.strip() == "land_transport_electric_share:":
            start_index.append(i)
        elif line.strip() == "land_transport_ice_share:":
            start_index.append(i)

    numb = next((i for i, num in enumerate(start_index) if num > first), None)
    start_index = start_index[numb : numb+ 3]

    j = 0
    # Modify the content accordingly
    for key in mapping_transport.keys():
        idx = start_index[j] + 1
        for col in transport_share.columns:
            if pd.notnull(transport_share.loc[key, col]):
                lines[idx] = f"      {col}: {transport_share.loc[key, col]:.3f}\n"
            idx += 1
        j += 1

    # naval sector
    mapping_navigation = {
        'H2': 'shipping_hydrogen_share',
        'MeOH': 'shipping_methanol_share',
        'Oil': 'shipping_oil_share',
    }
    
    # transport sector
    start_index = []
    for i, line in enumerate(lines):
        if line.strip() == "shipping_hydrogen_share:":
            start_index.append(i)
        elif line.strip() == "shipping_methanol_share:":
            start_index.append(i)
        elif line.strip() == "shipping_oil_share:":
            start_index.append(i)

    numb = next((i for i, num in enumerate(start_index) if num > first), None)
    start_index = start_index[numb : numb+ 3]

    j = 0
    # Modify the content accordingly
    for key in mapping_navigation.keys():
        idx = start_index[j] + 1
        for col in naval_share.columns:
            if pd.notnull(naval_share.loc[key, col]):
                lines[idx] = f"      {col}: {naval_share.loc[key, col]:.3f}\n"
            idx += 1
        j += 1
        
    # Write the modified content back to the file
    with open(output, 'w') as f:
        f.writelines(lines)

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

    scenario = snakemake.params.scenario_name[0]
    output = snakemake.input[0]
    write_to_scenario_yaml(output, scenario, transport_share, naval_share)