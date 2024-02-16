import pyam
import pandas as pd
import os
# Set USERNAME and PASSWORD for the Ariadne DB
pyam.iiasa.set_config(
    os.environ["IIASA_USERNAME"], 
    os.environ["IIASA_PASSWORD"],
)

model_df= pyam.read_iiasa(
    "ariadne_intern",
    model="Hybrid",
    scenario="8Gt_Bal_v3",
).timeseries()

df = model_df.loc["Hybrid", "8Gt_Bal_v3", "Deutschland"]

baseline_ksg = 1251
baseline_pypsa = 1052

## GHG target according to KSG
initial_years_ksg = pd.Series(
    index = [2020, 2025, 2030],
    data = [813, 643, 438],
)

later_years_ksg = pd.Series(
    index = [2035, 2040, 2045, 2050],
    data = [0.77, 0.88, 1.0, 1.0],
)

targets_ksg = pd.concat(
    [initial_years_ksg, (1 - later_years_ksg) * baseline_ksg],
)

## Compute nonco2 from Ariadne-Hybrid model

co2_ksg = (
    df.loc["Emissions|CO2 incl Bunkers","Mt CO2/yr"]  
    - df.loc["Emissions|CO2|Land-Use Change","Mt CO2-equiv/yr"]
    - df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
)

ghg_ksg = (
    df.loc["Emissions|Kyoto Gases","Mt CO2-equiv/yr"]
    - df.loc["Emissions|Kyoto Gases|Land-Use Change","Mt CO2-equiv/yr"]
    # No Kyoto Gas emissions for Bunkers recorded in Ariadne DB
)

nonco2 = ghg_ksg - co2_ksg

## PyPSA disregards nonco2 GHG emissions, but includes bunkers

targets_pypsa = (
    targets_ksg - nonco2 
    + df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
)

target_fractions_pypsa = (
    targets_pypsa.loc[targets_ksg.index] / baseline_pypsa
)

print("co2_budget_national:")

for year in target_fractions_pypsa.index:
    print("  ", year, ":", sep="")
    print("    DE:", target_fractions_pypsa[year].round(3))