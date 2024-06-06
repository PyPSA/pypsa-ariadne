# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script reads in data from the IIASA database to create the scenario.yaml file
import logging
logger = logging.getLogger(__name__)

import ruamel.yaml
from pathlib import Path
import pandas as pd
import os

def get_transport_shares(df, planning_horizons):
    # Get share of vehicles for transport sector - neglecting heavy duty vehicles
    total_transport = df.loc["Aladin v1", "Stock|Transportation|LDV"]
    tech_transport = df.loc["Aladin v1"].loc[[ 
        "Stock|Transportation|LDV|ICE",
        "Stock|Transportation|LDV|BEV",
        "Stock|Transportation|LDV|PHEV",
    ]]

    transport_share = tech_transport / total_transport
    transport_share = transport_share[planning_horizons]
    transport_share.set_index(pd.Index(["ICE", "BEV", "PHEV"]), inplace=True)

    # Get share of Navigation fuels from corresponding "Ariadne Leitmodell"
    total_navigation = \
        df.loc["REMIND-EU v1.1", "Final Energy|Bunkers|Navigation"] + \
        df.loc["Aladin v1", "Final Energy|Transportation|Domestic Navigation"]
    navigation_liquid = \
        df.loc["REMIND-EU v1.1", "Final Energy|Bunkers|Navigation|Liquids"] + \
        df.loc["Aladin v1", "Final Energy|Transportation|Domestic Navigation|Liquids"]
    
    navigation_h2 = df.loc["Aladin v1", "Final Energy|Transportation|Domestic Navigation|Hydrogen"]    

    h2_share = navigation_h2 / total_navigation
    liquid_share = navigation_liquid / total_navigation
    methanol_share = (1 - h2_share - liquid_share).round(6)
    
    naval_share = pd.concat(
            [liquid_share, h2_share, methanol_share]).set_index(
            pd.Index(["Oil", "H2", "MeOH"])
        )[planning_horizons]

    return transport_share, naval_share

def get_transport_growth(df, planning_horizons):
    # Aviation growth factor - using REMIND-EU v1.1 since Aladin v1 does not include bunkers
    aviation_model = "REMIND-EU v1.1"
    aviation = df.loc[aviation_model,"Final Energy|Bunkers|Aviation", "PJ/yr"]
    aviation_growth_factor = aviation / aviation[2020]

    # Transport growth factor - using REMIND until Aladin v1 uploads variables
    transport_model = "REMIND-EU v1.1"
    freight = df.loc[transport_model, "Energy Service|Transportation|Freight|Road", "bn tkm/yr"]
    person = df.loc[transport_model, "Energy Service|Transportation|Passenger|Road", "bn pkm/yr"]
    freight_PJ = df.loc[transport_model, "Final Energy|Transportation|Truck", "PJ/yr"]
    person_PJ = df.loc[transport_model, "Final Energy|Transportation|LDV", "PJ/yr"]
    
    transport_growth_factor = pd.Series()
    for year in planning_horizons:
        share = (person_PJ[year] / (person_PJ[year] + freight_PJ[year]))
        transport_growth_factor.loc[year] = share * (person[year] / person[2020]) + (1 - share) * (freight[year] / freight[2020])

    return aviation_growth_factor[planning_horizons], transport_growth_factor


def get_primary_steel_share(df, planning_horizons):
    # Get share of primary steel production
    model = "FORECAST v1.0"
    total_steel = df.loc[model, "Production|Steel"]
    primary_steel = df.loc[model, "Production|Steel|Primary"]
    
    primary_steel_share = primary_steel / total_steel
    primary_steel_share = primary_steel_share[planning_horizons]

    if model == "FORECAST v1.0" and planning_horizons[0] == 2020:
        logger.warning("FORECAST v1.0 does not have data for 2020. Using 2021 data for Production|Steel instead.")
        primary_steel_share[2020] = primary_steel[2021] / total_steel[2021]

    
    return primary_steel_share.set_index(pd.Index(["Primary_Steel_Share"]))

def get_co2_budget(df, source):
    # relative to the DE emissions in 1990 *including bunkers*; also
    # account for non-CO2 GHG and allow extra room for international
    # bunkers which are excluded from the national targets

    # Baseline emission in DE in 1990 in Mt as understood by the KSG and by PyPSA
    baseline_co2 = 1251
    baseline_pypsa = 1052
    if source == "KSG":
        ## GHG targets according to KSG
        initial_years_co2 = pd.Series(
            index = [2020, 2025, 2030],
            data = [813, 643, 438],
        )

        later_years_co2 = pd.Series(
            index = [2035, 2040, 2045, 2050],
            data = [0.77, 0.88, 1.0, 1.0],
        )

        targets_co2 = pd.concat(
            [initial_years_co2, (1 - later_years_co2) * baseline_co2],
        )
    elif source == "UBA":
        ## For Zielverfehlungsszenarien use UBA Projektionsbericht
        targets_co2 = pd.Series(
            index = [2020, 2025, 2030, 2035, 2040, 2045, 2050],
            data = [813, 655, 455, 309, 210, 169, 157],
        )
    else:
        raise ValueError("Invalid source for CO2 budget.")
    ## Compute nonco2 from Ariadne-Leitmodell (REMIND)

    co2 = (
        df.loc["Emissions|CO2 incl Bunkers","Mt CO2/yr"]  
        - df.loc["Emissions|CO2|Land-Use Change","Mt CO2-equiv/yr"]
        - df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
    )

    ghg = (
        df.loc["Emissions|Kyoto Gases","Mt CO2-equiv/yr"]
        - df.loc["Emissions|Kyoto Gases|Land-Use Change","Mt CO2-equiv/yr"]
        # No Kyoto Gas emissions for Bunkers recorded in Ariadne DB
    )

    nonco2 = ghg - co2

    ## PyPSA disregards nonco2 GHG emissions, but includes bunkers

    targets_pypsa = (
        targets_co2 - nonco2 
        + df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
    )

    target_fractions_pypsa = (
        targets_pypsa.loc[targets_co2.index] / baseline_pypsa
    )

    return target_fractions_pypsa.round(3)


def write_to_scenario_yaml(
        input, output, scenarios, df):
    # read in yaml file
    yaml = ruamel.yaml.YAML()
    file_path = Path(input)
    config = yaml.load(file_path)
    for scenario in scenarios:
        reference_scenario = config[scenario]["iiasa_database"]["reference_scenario"]
        co2_budget_source = config[scenario]["co2_budget_DE_source"]

        co2_budget_fractions = get_co2_budget(
            df.loc["REMIND-EU v1.1", reference_scenario],
            co2_budget_source
        )
        
        planning_horizons = [2020, 2025, 2030, 2035, 2040, 2045] # for 2050 we still need data

        transport_share, naval_share = get_transport_shares(
            df.loc[:, reference_scenario, :],
            planning_horizons,
        )
        
        aviation_demand_factor, land_transport_demand_factor = get_transport_growth(df.loc[:, reference_scenario, :], planning_horizons)

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

        config[scenario]["sector"] = {}
        for key, sector_mapping in mapping_transport.items():
            config[scenario]["sector"][sector_mapping] = {}
            for year in transport_share.columns:
                config[scenario]["sector"][sector_mapping][year] = round(transport_share.loc[key, year].item(), 4)

        for key, sector_mapping in mapping_navigation.items():
            config[scenario]["sector"][sector_mapping] = {}
            for year in naval_share.columns:
                config[scenario]["sector"][sector_mapping][year] = round(naval_share.loc[key, year].item(), 4)
        config[scenario]["sector"]["land_transport_demand_factor"] = {}
        config[scenario]["sector"]["aviation_demand_factor"] = {}
        for year in planning_horizons:
            config[scenario]["sector"]["aviation_demand_factor"][year] = round(aviation_demand_factor.loc[year].item(), 4)
            config[scenario]["sector"]["land_transport_demand_factor"][year] = round(land_transport_demand_factor.loc[year].item(), 4)

        st_primary_fraction = get_primary_steel_share(df.loc[:, reference_scenario, :], planning_horizons)
        
        config[scenario]["industry"] = {}
        config[scenario]["industry"]["St_primary_fraction"] = {}
        for year in st_primary_fraction.columns:
            config[scenario]["industry"]["St_primary_fraction"][year] = round(st_primary_fraction.loc["Primary_Steel_Share", year].item(), 4)
        config[scenario]["co2_budget_national"] = {}
        for year, target in co2_budget_fractions.items():
            config[scenario]["co2_budget_national"][year] = {}
            config[scenario]["co2_budget_national"][year]["DE"] = target

    # write back to yaml file
    yaml.dump(config, Path(output))



if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_scenarios")

    # Set USERNAME and PASSWORD for the Ariadne DB
    ariadne_db = pd.read_csv(
        snakemake.input.ariadne_database,
        index_col=["model", "scenario", "region", "variable", "unit"]
    )
    ariadne_db.columns = ariadne_db.columns.astype(int)

    df = ariadne_db.loc[
        :, 
        :,
        "Deutschland"]
    
    scenarios = snakemake.params.scenarios

    input = snakemake.input.scenario_yaml
    output = snakemake.output.scenario_yaml

    # for scenario in scenarios:
    write_to_scenario_yaml(
        input, output, scenarios, df)
