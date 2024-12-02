# -*- coding: utf-8 -*-
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
import sys
import os
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

    fraction_DE = get(snakemake.params.future_DE, year)
    fraction_EU = get(snakemake.params.future_EU, year)

    intermediate_sector_ratios = {}
    # TODO: ideally I would just change the data for Germany without writing the file again
    for ct, group in today_sector_ratios.T.groupby(level=0):
        if ct == "DE":
            fraction_future = fraction_DE
        else:
            fraction_future = fraction_EU
        today_sector_ratios_ct = group.droplevel(0).T.reindex_like(future_sector_ratios)
        missing_mask = today_sector_ratios_ct.isna().all()
        today_sector_ratios_ct.loc[:, missing_mask] = future_sector_ratios.loc[
            :, missing_mask
        ]
        today_sector_ratios_ct.loc[:, ~missing_mask] = today_sector_ratios_ct.loc[
            :, ~missing_mask
        ].fillna(future_sector_ratios)
        intermediate_sector_ratios[ct] = (
            today_sector_ratios_ct * (1 - fraction_future)
            + future_sector_ratios * fraction_future
        )

    intermediate_sector_ratios = pd.concat(intermediate_sector_ratios, axis=1)

    intermediate_sector_ratios.to_csv(snakemake.output.sector_ratios_modified)


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_sector_ratios",
            planning_horizons="2045",
        )

    year = int(snakemake.wildcards.planning_horizons)

    build_industry_sector_ratios_intermediate()
