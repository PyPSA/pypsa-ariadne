# -*- coding: utf-8 -*-
import logging

import pandas as pd
from _helpers import configure_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_existing_heating",
            run="KN2045_Bal_v4",
        )

    configure_logging(snakemake)

    leitmodell = snakemake.params.leitmodelle["buildings"]
    logger.info(f"Using {leitmodell} for heating demand modification.")

    existing_heating = pd.read_csv(snakemake.input.existing_heating, index_col=0)

    ariadne = pd.read_csv(
        snakemake.input.ariadne,
        index_col=["model", "scenario", "region", "variable", "unit"],
    ).loc[
        leitmodell,
        snakemake.params.fallback_reference_scenario,
        "Deutschland",
        :,
        "million",
    ]

    logger.info(f"Heating demand before modification:{existing_heating.loc['Germany']}")

    mapping = {
        "gas boiler": "Gas Boiler",
        "oil boiler": "Oil Boiler",
        "air heat pump": "Heat Pump|Electrical|Air",
        "ground heat pump": "Heat Pump|Electrical|Ground",
        "biomass boiler": "Biomass Boiler",
    }

    new_values = pd.Series()

    year = "2020"
    for tech in mapping:
        stock = ariadne.at[
            f"Stock|Space Heating|{mapping[tech]}",
            year,
        ]

        peak = (
            stock
            * existing_heating.loc["Germany"].sum()
            / ariadne.at[f"Stock|Space Heating", year]
        )
        new_values[tech] = peak

    if any(new_values.isna()):
        logger.warning(
            f"Missing values for {new_values[new_values.isna()].index.to_list()}. Switching to hard coded values from a previous REMod run."
        )

        total_stock = 23.28  # million
        existing_factor = existing_heating.loc["Germany"].sum() / total_stock

        new_values["gas boiler"] = 11.44
        new_values["oil boiler"] = 5.99
        new_values["air heat pump"] = 0.38
        new_values["ground heat pump"] = 0.38
        new_values["biomass boiler"] = 2.8

        logger.info(new_values)
        logger.warning(f"Total stock: {total_stock}, New stock: {new_values.sum()}")
        logger.warning(f"District heating is not correctly accounted for in the new stock.")
        new_values *= existing_factor

    for tech, peak in new_values.items():
        existing_heating.at["Germany", tech] = peak

    logger.info(
        f"Heating demand after modification: {existing_heating.loc['Germany']}"
    )

    existing_heating.to_csv(snakemake.output.existing_heating)
