

import pandas as pd

# leitmodell for steel and cement demand
leitmodell="FORECAST v1.0"

year = snakemake.input.industrial_production_per_country_tomorrow.split("_")[-1].split(".")[0]

existing_industry = pd.read_csv(snakemake.input.industrial_production_per_country_tomorrow, index_col=0)

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
existing_industry.loc["DE", "Cement"] = ariadne.loc["Production|Non-Metallic Minerals|Cement", year]

# get steel ratios from existing_industry
steel = existing_industry.loc["DE", ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]]
ratio = steel/steel.sum()

# multiply with steel production including primary and secondary steel since distinguishing is taken care of later
existing_industry.loc["DE", ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]] = ratio * ariadne.loc["Production|Steel", year]

print(
    "German demand after modification", 
    existing_industry.loc["DE", ["Cement", "Electric arc", "Integrated steelworks", "DRI + Electric arc"]], sep="\n")

existing_industry.to_csv(snakemake.output.industrial_production_per_country_tomorrow)
