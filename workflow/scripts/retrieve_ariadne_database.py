# -*- coding: utf-8 -*-
import logging

import pyam
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_ariadne_database")

    configure_logging(snakemake)
    logger.info(
        f"Retrieving from IIASA database {snakemake.params.db_name}\nmodels {list(snakemake.params.leitmodelle.values())}\nscenarios {snakemake.params.scenarios}"
    )

    db = pyam.read_iiasa(
        snakemake.params.db_name,
        model=snakemake.params.leitmodelle.values(),
        scenario=snakemake.params.scenarios,
        # Download only the most recent iterations of scenarios
    )

    logger.info(f"Successfully retrieved database.")
    db.timeseries().to_csv(snakemake.output.data)
