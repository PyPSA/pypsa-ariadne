import logging

import pandas as pd
import numpy as np

import os
import sys

paths = ["workflow/submodules/pypsa-eur/scripts", "../submodules/pypsa-eur/scripts"]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))
from prepare_sector_network import prepare_costs

logger = logging.getLogger(__name__)

def calculate_annuity(invest, fom, lifetime, r):
    """
    Calculate annuity based on EAC.

    invest - investment
    fom    - annual FOM in percentage of investment
    lifetime - lifetime of investment in years
    r      - discount rate in percent
    """

    r = r / 100.0

    annuity_factor = r / (1.0 - 1.0 / (r + 1.0) ** (lifetime))

    return (annuity_factor + fom / 100.0) * invest

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_import_costs",
            planning_horizons="2030",
            run="no_import_he",
        )

    logger.info("Modfiying import costs from TRACE data.")

    # read in current import costs
    import_costs = pd.read_csv(
        snakemake.input.import_costs, delimiter=";", keep_default_na=False
    )
    # read in new cost data
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        1)

    # corresponding to TRACE
    invest_factor = 1000
    fom = 4
    lifetime = 25
    r = 5
    invest = invest_factor * 450
    old_annuity = calculate_annuity(invest, fom, lifetime, r)
    # 2015 € in 2020 €
    inflation_rates = [0.0273, 0.0273, 0.0325, 0.0365, 0.0351] #, 0.0324]
    product_of_factors = 1
    for rate in inflation_rates:
        product_of_factors *= (1 + rate)

    # Calculate deflation factor
    deflation_factor = 1 / product_of_factors

    # new annuity
    invest = costs.at["electrolysis", "investment"] # invest factor already included
    fom = costs.at["electrolysis", "FOM"]
    lifetime = costs.at["electrolysis", "lifetime"]
    new_annuity = calculate_annuity(invest, fom, lifetime, r) * deflation_factor

    # get volume, total system costs and electrolysis costs
    new_volumes = pd.DataFrame()
    i = 0
    for exp in import_costs.exporter.unique():
        data = import_costs[import_costs.exporter == exp]
        for imp in data.importer.unique():
            for carrier in data[data.importer==imp].esc.unique():
                if carrier == "hvdc-to-elec":
                    continue
                new_volumes.loc[i, "esc"] = carrier
                new_volumes.loc[i, "exporter"] = exp
                new_volumes.loc[i, "importer"] = imp
                
                total_costs = data[(data.importer == imp) & (data.esc == carrier) & (data.subcategory=="Total system cost")].value
                electrolysis_costs = data[(data.importer == imp) & (data.esc == carrier) & (data.category == "cost") & (data.subcategory=="electrolysis (exp)")].value
                electrolysis_cap = data[(data.importer == imp) & (data.esc == carrier) & (data.category == "installed capacity") & (data.subcategory=="electrolysis (exp)")].value

                new_total = total_costs.values[0] - electrolysis_costs.values[0] + electrolysis_cap.values[0]*new_annuity

                volume = data[(data.importer == imp) & (data.esc == carrier) & (data.subcategory=="Total demand")].value
                new_volumes.loc[i, "marginal_cost"] = new_total / volume.values[0]
                new_volumes.loc[i, "marginal_cost"] /= deflation_factor
                i = i + 1
    
    # write new costs
    new_volumes.to_csv(snakemake.output.import_costs)