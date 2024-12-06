#-*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Overwrite the sector ratios for Germany to represent climate neutrality in 2045 already compared to 2050 for the rest of Europe.
Relevant Settings
-----------------
.. code:: yaml
    industry:
        sector_ratios_fraction_future_DE:
Inputs
------
- ``resources/industry_sector_ratios.csv``
- ``resources/industrial_energy_demand_per_country_today.csv``
- ``resources/industrial_production_per_country.csv``
Outputs
-------
- ``resources/industry_sector_ratios_{planning_horizons}-modified.csv``
"""
import os
import sys

paths = ["workflow/submodules/pypsa-eur/scripts", "../submodules/pypsa-eur/scripts"]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

import numpy as np
import pandas as pd
from prepare_sector_network import get


def build_industry_sector_ratios_intermediate():

    # in TWh/a
    demand = pd.read_csv(
        snakemake.input.industrial_energy_demand_per_country_today,
        header=[0, 1],
        index_col=0,
    )

    # in Mt/a
    production = (
        pd.read_csv(snakemake.input.industrial_production_per_country, index_col=0)
        / 1e3
    ).stack()
    production.index.names = [None, None]

    # in MWh/t
    future_sector_ratios = pd.read_csv(
        snakemake.input.industry_sector_ratios, index_col=0
    )

    today_sector_ratios = demand.div(production, axis=1).replace([np.inf, -np.inf], 0)

    today_sector_ratios.dropna(how="all", axis=1, inplace=True)

    rename = {
        "waste": "biomass",
        "electricity": "elec",
        "solid": "coke",
        "gas": "methane",
        "other": "biomass",
        "liquid": "naphtha",
    }
    today_sector_ratios = today_sector_ratios.rename(rename).groupby(level=0).sum()

    # custom DE pathway
    fraction_DE = get(snakemake.params.future_DE, year)

    intermediate_sector_ratios_DE = {}

    DE_sector_ratios = today_sector_ratios.loc[:, "DE"].reindex_like(
        future_sector_ratios
    )
    missing_mask = DE_sector_ratios.isna().all()
    DE_sector_ratios.loc[:, missing_mask] = future_sector_ratios.loc[:, missing_mask]
    DE_sector_ratios.loc[:, ~missing_mask] = DE_sector_ratios.loc[
        :, ~missing_mask
    ].fillna(future_sector_ratios)
    intermediate_sector_ratios_DE["DE"] = (
        DE_sector_ratios * (1 - fraction_DE) + future_sector_ratios * fraction_DE
    )
    # make dictionary to dataframe
    intermediate_sector_ratios_DE = pd.concat(intermediate_sector_ratios_DE, axis=1)

    # read in original sector ratios
    intermediate_sector_ratios = pd.read_csv(
        snakemake.input.sector_ratios, header=[0, 1], index_col=0
    )

    # update DE sector ratios
    intermediate_sector_ratios.loc[:, "DE"] = intermediate_sector_ratios_DE["DE"].values

    intermediate_sector_ratios.to_csv(snakemake.output.sector_ratios_modified)


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_sector_ratios",
            planning_horizons="2045",
            run="KN2045_Bal_v4",
        )

    year = int(snakemake.wildcards.planning_horizons)

    build_industry_sector_ratios_intermediate()
