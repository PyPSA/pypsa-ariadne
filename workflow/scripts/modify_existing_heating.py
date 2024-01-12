

import pandas as pd


existing_heating = pd.read_csv(snakemake.input.existing_heating,
                               index_col=0)

ariadne = pd.read_csv(snakemake.input.ariadne,
                      index_col=0)

print("before",existing_heating.loc["Germany"])

mapping = {"gas boiler" : "Gas Boiler",
                  "oil boiler" : "Oil Boiler",
                  "air heat pump" : "Heat Pump|Electrical|Air",
                  "ground heat pump" : "Heat Pump|Electrical|Ground",
                  "biomass boiler" : "Biomass Boiler"}

year = "2020"
for tech in mapping:
    stock = ariadne.at[f"Stock|Space Heating|{mapping[tech]}",year]
    peak = stock*existing_heating.loc["Germany"].sum()/ariadne.at[f"Stock|Space Heating",year]
    existing_heating.at["Germany",tech] = peak



print("before",existing_heating.loc["Germany"])


existing_heating.to_csv(snakemake.output.existing_heating)
