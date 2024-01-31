


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

    for ct in snakemake.config["new_decentral_fossil_boiler_ban"]:
        ban_year = int(snakemake.config["new_decentral_fossil_boiler_ban"][ct])
        if ban_year < year:
            logger.info(f"For year {year} in {ct} implementing ban on new decentral oil & gas boilers from {ban_year}")
            links = n.links.index[(n.links.index.str[:2] == ct) & (n.links.index.str.contains("gas boiler") ^ n.links.index.str.contains("oil boiler")) & n.links.p_nom_extendable & ~n.links.index.str.contains("urban central")]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)

def coal_generation_ban(n):

    year = int(snakemake.wildcards.planning_horizons)

    for ct in snakemake.config["coal_generation_ban"]:
        ban_year = int(snakemake.config["coal_generation_ban"][ct])
        if ban_year < year:
            logger.info(f"For year {year} in {ct} implementing coal and lignite ban from {ban_year}")
            links = n.links.index[(n.links.index.str[:2] == ct) & n.links.carrier.isin(["coal","lignite"])]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)

def add_wasserstoff_kernnetz(n, wkn, costs):

    logger.info("adding wasserstoff kernnetz")

    investment_year = int(snakemake.wildcards.planning_horizons)

    # get last planning horizon
    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    i = planning_horizons.index(int(snakemake.wildcards.planning_horizons))
    
    if i != 0:
        last_investment_year = int(planning_horizons[i - 1])
    else:
        last_investment_year = 2015

    wkn_all = wkn.copy()
    # use only pipes which are present between the current year and the last investment period
    wkn.query("build_year > @last_investment_year & build_year <= @investment_year", inplace=True)

    if not wkn.empty:

        # add kernnetz to network
        n.madd(
            "Link",
            wkn.index + f"-{investment_year}-kernnetz",
            bus0=wkn.bus0.values + " H2",
            bus1=wkn.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=False,
            p_nom=wkn.p_nom.values,
            build_year=wkn.build_year.values,
            length=wkn.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * wkn.length.values,
            carrier="H2 pipeline (kernnetz)",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

        # add reversed pipes and losses
        lossy_bidirectional_links(n, "H2 pipeline (kernnetz)", snakemake.config["sector"]["transmission_efficiency"]["H2 pipeline"])

        # reverte carrier change
        n.links.loc[n.links.carrier == "H2 pipeline (kernnetz)", "carrier"] = "H2 pipeline"

        if investment_year <= 2030:
            # assume that only pipelines from kernnetz are build (within Germany): make pipes within Germany not extendable and all others extendable (but only from current year)
            n.links.loc[(n.links.carrier == "H2 pipeline") & n.links.p_nom_extendable,"p_nom_extendable"] = n.links.loc[(n.links.carrier == "H2 pipeline") & n.links.p_nom_extendable,:].apply(lambda row: False if (row.bus0[:2] == "DE") & (row.bus1[:2] == "DE") else True, axis=1)

        # from 2030 onwards  all pipes are extendable (except from the ones the model build up before and the kernnetz lines)
            
        # add reversed pipes
        def add_reversed_pipes(df):
            df_rev = df.copy()
            df_rev.index = df_rev.index + "_rev"
            df_rev["bus0"] = df.bus1.values
            df_rev["bus1"] = df.bus0.values
            result = pd.concat([df,df_rev])
            return result
            
        def reduce_capacity(df, pipes, carrier, origin="removed_gas_cap", target="p_nom", efficiency=1):
            """
            Reduce the capacity of pipes in a dataframe based on specified criteria.

            Args:
                df (DataFrame): The input dataframe containing the pipe data to be modified.
                pipes (DataFrame): The dataframe containing the data how to midify.
                carrier (str): The carrier of the pipes.
                origin (str, optional): The column name in `df` representing the original capacity of the pipes. Defaults to "removed_gas_cap".
                target (str, optional): The column name in `df` representing the target capacity to be modified. Defaults to "p_nom".
                efficiency (float, optional): The efficiency factor to reduce the capacity. Defaults to 1.

            Returns:
                DataFrame: The modified dataframe with reduced pipe capacities.
            """

            result = df.copy()

            for i, pipe in pipes.iterrows():
                cut = pipe[origin] * efficiency
                match_i = 0
                while cut > 0:
                    match = result[(result.bus0 == pipe.bus0 + " " + carrier) & (result.bus1 == pipe.bus1 + " " + carrier)]
                    if (match.empty) | (match_i >= len(match)):
                        break
                    target_value = match.iloc[match_i][target]
                    if target_value <= cut:
                        result.loc[match.index[match_i], target] -= target_value
                        cut -= target_value
                        match_i += 1
                    else:
                        result.loc[match.index[match_i], target] -= cut
                        cut = 0
                        match_i += 1
            return result

        # reduce the gas network capacity of retrofitted lines from kernnetz which is build in the current period
        if snakemake.config["sector"]["gas_network"]:

            gas_pipes = n.links[(n.links.carrier == "gas pipeline")][["bus0", "bus1", "p_nom"]].copy()
            res_gas_pipes = reduce_capacity(gas_pipes, 
                                            add_reversed_pipes(wkn), 
                                            carrier="gas", 
                                            origin="removed_gas_cap", 
                                            target="p_nom", 
                                            efficiency=1)
            n.links.loc[(n.links.carrier == "gas pipeline"),"p_nom"] = res_gas_pipes["p_nom"]

        # reduce H2 retrofitting potential from gas network for all kernnetz pipelines which are being build in total (more conservative approach)
        if snakemake.config["sector"]["H2_retrofit"]:

            h2_pipes_retrofitted = n.links[(n.links.carrier == "H2 pipeline retrofitted") & (n.links.index.str.contains(str(investment_year)))][["bus0", "bus1", "p_nom_max"]].copy()
            res_h2_pipes_retrofitted = reduce_capacity(h2_pipes_retrofitted, 
                                                       add_reversed_pipes(wkn_all), 
                                                       carrier="H2", 
                                                       origin="removed_gas_cap", 
                                                       target="p_nom_max", 
                                                       efficiency=snakemake.config["sector"]["H2_retrofit_capacity_per_CH4"])
            n.links.loc[(n.links.carrier == "H2 pipeline retrofitted") & (n.links.index.str.contains(str(investment_year))), "p_nom_max"] = res_h2_pipes_retrofitted["p_nom_max"]


if __name__ == "__main__":

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

    if snakemake.config["wasserstoff_kernnetz"]["enable"]:
        fn = snakemake.input.wkn
        wkn = pd.read_csv(fn, index_col=0)
        add_wasserstoff_kernnetz(n, wkn, costs)

    n.export_to_netcdf(snakemake.output.network)
