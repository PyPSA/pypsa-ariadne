import logging

import pandas as pd

import pypsa

import os
import sys

logger = logging.getLogger(__name__)

paths = ["workflow/submodules/pypsa-eur/scripts", "../submodules/pypsa-eur/scripts"]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))
from prepare_sector_network import (
    prepare_costs,
    lossy_bidirectional_links,
)

from add_electricity import load_costs


def first_technology_occurrence(n):
    """
    Sets p_nom_extendable to false for carriers with configured first occurrence
    if investment year is before configured year.
    """

    for c, carriers in snakemake.params.technology_occurrence.items():
        for carrier, first_year in carriers.items():
            if int(snakemake.wildcards.planning_horizons) < first_year:
                logger.info(f"{carrier} not extendable before {first_year}.")
                n.df(c).loc[n.df(c).carrier == carrier, "p_nom_extendable"] = False
  

def fix_new_boiler_profiles(n):

    logger.info("Forcing boiler profiles for new ones")

    decentral_boilers = n.links.index[n.links.carrier.str.contains("boiler")
                                      & ~n.links.carrier.str.contains("urban central")
                                      & n.links.p_nom_extendable]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers,"bus1"]
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(n.loads_t.p_set[boiler_loads].max(),axis=1)
    boiler_profiles_pu.columns = decentral_boilers

    for attr in ["p_min_pu","p_max_pu"]:
        n.links_t[attr] = pd.concat([n.links_t[attr], boiler_profiles_pu],axis=1)
        logger.info(f"new boiler profiles:\n{n.links_t[attr][decentral_boilers]}")


def remove_old_boiler_profiles(n):
    """Removed because this is handled in additional_functionality.
    This removes p_min/max_pu constraints added in previous years
    and carried over by add_brownfield.
    """

    logger.info("Removing p_min/max_pu constraints on old boiler profiles")


    decentral_boilers = n.links.index[n.links.carrier.str.contains("boiler")
                                      & ~n.links.carrier.str.contains("urban central")
                                      & ~n.links.p_nom_extendable]

    for attr in ["p_min_pu","p_max_pu"]:
        to_drop = decentral_boilers.intersection(n.links_t[attr].columns)
        logger.info(f"Dropping {to_drop} from n.links_t.{attr}")
        n.links_t[attr].drop(to_drop, axis=1, inplace=True)


def new_boiler_ban(n):

    year = int(snakemake.wildcards.planning_horizons)

    for ct in snakemake.params.fossil_boiler_ban:
        ban_year = int(snakemake.params.fossil_boiler_ban[ct])
        if ban_year < year:
            logger.info(f"For year {year} in {ct} implementing ban on new decentral oil & gas boilers from {ban_year}")
            links = n.links.index[(n.links.index.str[:2] == ct) & (n.links.index.str.contains("gas boiler") ^ n.links.index.str.contains("oil boiler")) & n.links.p_nom_extendable & ~n.links.index.str.contains("urban central")]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)

def coal_generation_ban(n):

    year = int(snakemake.wildcards.planning_horizons)

    for ct in snakemake.params.coal_ban:
        ban_year = int(snakemake.params.coal_ban[ct])
        if ban_year < year:
            logger.info(f"For year {year} in {ct} implementing coal and lignite ban from {ban_year}")
            links = n.links.index[(n.links.index.str[:2] == ct) & n.links.carrier.isin(["coal","lignite"])]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)
            
def nuclear_generation_ban(n):

    year = int(snakemake.wildcards.planning_horizons)

    for ct in snakemake.params.nuclear_ban:
        ban_year = int(snakemake.params.nuclear_ban[ct])
        if ban_year < year:
            logger.info(f"For year {year} in {ct} implementing nuclear ban from {ban_year}")
            links = n.links.index[(n.links.index.str[:2] == ct) & n.links.carrier.isin(["nuclear"])]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)            


def add_reversed_pipes(df):
    df_rev = df.copy().rename({"bus0": "bus1", "bus1": "bus0"}, axis=1)
    df_rev.index = df_rev.index + "-reversed"
    return pd.concat([df, df_rev], sort=False)


def reduce_capacity(targets, origins, carrier, origin_attr="removed_gas_cap", target_attr="p_nom", conversion_rate=1):
    """
    Reduce the capacity of pipes in a dataframe based on specified criteria.

    Args:
        target (DataFrame): The dataframe containing pipelines from which to reduce capacitiy.
        origin (DataFrame): The dataframe containing data about added pipelines.
        carrier (str): The carrier of the pipelines.
        origin_attr (str, optional): The column name in `origin` representing the original capacity of the pipelines. Defaults to "removed_gas_cap".
        target_attr (str, optional): The column name in `target` representing the target capacity to be modified. Defaults to "p_nom".
        conversion_rate (float, optional): The conversion rate to reduce the capacity. Defaults to 1.

    Returns:
        DataFrame: The modified dataframe with reduced pipe capacities.
    """

    targets = targets.copy()

    def apply_cut(row):
        match = targets[
            (targets.bus0 == row.bus0 + " " + carrier) & 
            (targets.bus1 == row.bus1 + " " + carrier)
        ].sort_index()
        cut = row[origin_attr] * conversion_rate
        for idx, target_row in match.iterrows():
            if cut <= 0:
                break
            target_value = target_row[target_attr]
            reduction = min(target_value, cut)
            targets.at[idx, target_attr] -= reduction
            cut -= reduction

    origins.apply(apply_cut, axis=1)
    return targets


def add_wasserstoff_kernnetz(n, wkn, costs):

    logger.info("adding wasserstoff kernnetz")

    investment_year = int(snakemake.wildcards.planning_horizons)

    # get previous planning horizon
    planning_horizons = snakemake.params.planning_horizons
    i = planning_horizons.index(int(snakemake.wildcards.planning_horizons))
    previous_investment_year = int(planning_horizons[i - 1]) if i != 0 else 2015

    # use only pipes added since the previous investment period
    wkn_new = wkn.query("build_year > @previous_investment_year & build_year <= @investment_year")

    if not wkn_new.empty:

        names = wkn_new.index + f"-kernnetz-{investment_year}"

        # add kernnetz to network
        n.madd(
            "Link",
            names,
            bus0=wkn_new.bus0.values + " H2",
            bus1=wkn_new.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=False,
            p_nom=wkn_new.p_nom.values,
            build_year=wkn_new.build_year.values,
            length=wkn_new.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * wkn_new.length.values,
            carrier="H2 pipeline (Kernnetz)",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

        # add reversed pipes and losses
        losses = snakemake.params.H2_transmission_efficiency
        lossy_bidirectional_links(n, "H2 pipeline (Kernnetz)", losses, subset=names)

        # reduce the gas network capacity of retrofitted lines from kernnetz
        # which is build in the current period
        gas_pipes = n.links.query("carrier == 'gas pipeline'")
        if not gas_pipes.empty:
            res_gas_pipes = reduce_capacity(
                gas_pipes,
                add_reversed_pipes(wkn_new),
                carrier="gas",
            )
            n.links.loc[n.links.carrier == "gas pipeline", "p_nom"] = res_gas_pipes["p_nom"]

    # reduce H2 retrofitting potential from gas network for all kernnetz
    # pipelines which are being build in total (more conservative approach)
    if not wkn.empty and snakemake.params.H2_retrofit:

        conversion_rate = snakemake.params.H2_retrofit_capacity_per_CH4

        retrofitted_b = (
            n.links.carrier == "H2 pipeline retrofitted"
        ) & n.links.index.str.contains(str(investment_year))
        h2_pipes_retrofitted = n.links.loc[retrofitted_b]

        if not h2_pipes_retrofitted.empty:
            res_h2_pipes_retrofitted = reduce_capacity(
                h2_pipes_retrofitted,
                add_reversed_pipes(wkn),
                carrier="H2",
                target_attr="p_nom_max",
                conversion_rate=conversion_rate,
            )
            n.links.loc[retrofitted_b, "p_nom_max"] = res_h2_pipes_retrofitted["p_nom_max"]

    if investment_year <= 2030:
        # assume that only pipelines from kernnetz are built (within Germany):
        # make pipes within Germany not extendable and all others extendable (but only from current year)
        to_fix = (
            n.links.bus0.str.startswith("DE")
            & n.links.bus1.str.startswith("DE")
            & n.links.carrier.isin(["H2 pipeline", "H2 pipeline retrofitted"])
        )
        n.links.loc[to_fix, "p_nom_extendable"] = False

    # from 2030 onwards all pipes are extendable (except from the ones the model build up before and the kernnetz lines)

def unravel_oilbus(n):
    """
    Unravel European oil bus to enable energy balances for import of oil products.

    """
    logger.info("Unraveling oil bus")
    # add buses
    n.add("Bus", "DE", location="DE", x=10.5, y=51.2, carrier="none")
    n.add("Bus", "DE oil", location="DE", x=10.5, y=51.2, carrier="oil")
    n.add("Bus", "DE renewable oil", location="DE", x=10.5, y=51.2, carrier="renewable oil")
    n.add(
        "Bus", 
        "EU renewable oil", 
        location="EU",
        x=n.buses.loc["EU","x"],
        y=n.buses.loc["EU","y"],
        carrier="renewable oil"
    )

    # add one generator for DE oil
    n.add("Generator",
          name="DE oil",
          bus="DE oil",
          carrier="oil",
          p_nom_extendable=True,
          marginal_cost=n.generators.loc["EU oil"].marginal_cost,
          )
    
    # change links from EU oil to DE oil
    german_oil_links = n.links[(n.links.bus0=="EU oil") & (n.links.index.str.contains("DE"))].index
    german_FT_links = n.links[(n.links.bus1=="EU oil") & (n.links.index.str.contains("DE"))].index
    n.links.loc[german_oil_links, "bus0"] = "DE oil"
    n.links.loc[german_FT_links, "bus1"] = "DE renewable oil"

    # change FT links in rest of Europe
    europ_FT_links = n.links[n.links.bus1=="EU oil"].index
    n.links.loc[europ_FT_links, "bus1"] = "EU renewable oil"

    # add links between oil buses
    n.madd(
        "Link",
        ["EU renewable oil -> DE oil", "EU renewable oil -> EU oil", "DE renewable oil -> DE oil", "DE renewable oil -> EU oil"],
        bus0=["EU renewable oil", "EU renewable oil", "DE renewable oil", "DE renewable oil"],
        bus1=["DE oil", "EU oil", "DE oil", "EU oil"],
        carrier="renewable oil",
        p_nom=1e6,
        p_min_pu=0,
    )

    # add stores
    n.add("Store",
          "DE oil Store",
          bus="DE oil",
          carrier="oil",
          e_nom_extendable=True,
          e_cyclic=True,
          capital_cost=0.02,
          )


def transmission_costs_from_modified_cost_data(n, costs, length_factor=1.0):
    # copying the the function update_transmission_costs from add_electricity
    # slight change to the function so it works in modify_prenetwork
    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return
    # get all DC links that are not the reverse links
    dc_b = (n.links.carrier == "DC") & ~(n.links.index.str.contains("reverse"))

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"]
            * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_prenetwork",
            simpl="",
            clusters=22,
            opts="",
            ll="v1.2",
            sector_opts="365H-T-H-B-I-A-solar+p3-linemaxext15",
            planning_horizons="2040",
            run="KN2045_H2_v4"
        )

    logger.info("Adding Ariadne-specific functionality")


    n = pypsa.Network(snakemake.input.network)
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        nyears,
    )

    new_boiler_ban(n)

    fix_new_boiler_profiles(n)

    remove_old_boiler_profiles(n)

    coal_generation_ban(n)
    
    nuclear_generation_ban(n)

    first_technology_occurrence(n)

    if not snakemake.config["run"]["debug_unravel_oilbus"]:
        unravel_oilbus(n)

    if snakemake.params.enable_kernnetz:
        fn = snakemake.input.wkn
        wkn = pd.read_csv(fn, index_col=0)
        add_wasserstoff_kernnetz(n, wkn, costs)
        n.links.reversed = n.links.reversed.astype(float)

    costs_loaded = load_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        snakemake.params.max_hours,
        nyears,
    )

    # change to NEP21 costs
    transmission_costs_from_modified_cost_data(n, costs_loaded, snakemake.params.length_factor)

    n.export_to_netcdf(snakemake.output.network)
