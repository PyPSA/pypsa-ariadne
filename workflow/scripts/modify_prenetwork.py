


import logging

import pandas as pd

import pypsa

logger = logging.getLogger(__name__)



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

    logger.info(f"Implementing ban on new decentral gas & oil boilers (if there are any)")

    year = int(snakemake.wildcards.planning_horizons)

    logger.info(f"Current year is {year}")

    for ct in snakemake.config["new_decentral_fossil_boiler_ban"]:
        ban_year = int(snakemake.config["new_decentral_fossil_boiler_ban"][ct])
        logger.info(f"{ct} has a new gas/oil boiler ban from {ban_year}")
        if ban_year < year:
            logger.info(f"Implementing ban in this network")
            links = n.links.index[(n.links.index.str[:2] == ct) & (n.links.index.str.contains("gas boiler") ^ n.links.index.str.contains("oil boiler")) & n.links.p_nom_extendable & ~n.links.index.str.contains("urban central")]
            logger.info(f"Dropping {links}")
            n.links.drop(links,
                         inplace=True)

if __name__ == "__main__":

    logger.info("Adding Ariadne-specific functionality")


    n = pypsa.Network(snakemake.input.network)

    new_boiler_ban(n)

    fix_new_boiler_profiles(n)

    remove_old_boiler_profiles(n)

    n.export_to_netcdf(snakemake.output.network)
