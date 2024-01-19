


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
        n.links_t[attr][decentral_boilers] = boiler_profiles_pu
        print(n.links_t[attr][decentral_boilers])


if __name__ == "__main__":

    logger.info("Adding Ariadne-specific functionality")


    n = pypsa.Network(snakemake.input.network)

    fix_new_boiler_profiles(n)

    n.export_to_netcdf(snakemake.output.network)
