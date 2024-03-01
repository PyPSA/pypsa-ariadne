
import logging

import pandas as pd
from prepare_sector_network import determine_emission_sectors

from xarray import DataArray

logger = logging.getLogger(__name__)


def add_min_limits(n, snapshots, investment_year, config):

    for c in n.iterate_components(config["limits_min"]):
        logger.info(f"Adding minimum constraints for {c.list_name}")

        for carrier in config["limits_min"][c.name]:

            for ct in config["limits_min"][c.name][carrier]:
                limit = 1e3*config["limits_min"][c.name][carrier][ct][investment_year]

                logger.info(f"Adding constraint on {c.name} {carrier} capacity in {ct} to be greater than {limit} MW")

                existing_index = c.df.index[(c.df.index.str[:2] == ct) & (c.df.carrier.str[:len(carrier)] == carrier) & ~c.df.p_nom_extendable]
                extendable_index = c.df.index[(c.df.index.str[:2] == ct) & (c.df.carrier.str[:len(carrier)] == carrier) & c.df.p_nom_extendable]

                existing_capacity = c.df.loc[existing_index, "p_nom"].sum()

                logger.info(f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} MW")

                p_nom = n.model[c.name + "-p_nom"].loc[extendable_index]

                lhs = p_nom.sum()

                cname = f"capacity_minimum-{ct}-{c.name}-{carrier.replace(' ','-')}"

                n.model.add_constraints(
                    lhs >= limit - existing_capacity, name=f"GlobalConstraint-{cname}"
                )
                n.add(
                    "GlobalConstraint",
                    cname,
                    constant=limit,
                    sense=">=",
                    type="",
                    carrier_attribute="",
                )


def h2_import_limits(n, snapshots, investment_year, config):

    for ct in config["h2_import_max"]:
        limit = config["h2_import_max"][ct][investment_year]*1e6

        logger.info(f"limiting H2 imports in {ct} to {limit/1e6} TWh/a")

        incoming = n.links.index[(n.links.carrier == "H2 pipeline") & (n.links.bus0.str[:2] != ct) & (n.links.bus1.str[:2] == ct)]
        outgoing = n.links.index[(n.links.carrier == "H2 pipeline") & (n.links.bus0.str[:2] == ct) & (n.links.bus1.str[:2] != ct)]

        incoming_p = (n.model["Link-p"].loc[:, incoming]*n.snapshot_weightings.generators).sum()
        outgoing_p = (n.model["Link-p"].loc[:, outgoing]*n.snapshot_weightings.generators).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_import_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit, name=f"GlobalConstraint-{cname}"
        )
        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def add_co2limit_country(n, limit_countries, snakemake, investment_year):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_national'.

    Parameters
    ----------
    n : pypsa.Network
    limit_countries : dict
    snakemake: snakemake object
    """
    logger.info(f"Adding CO2 budget limit for each country as per unit of 1990 levels")

    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    sectors = determine_emission_sectors(n.config['sector'])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in limit_countries:
        limit = co2_total_totals[ct]*limit_countries[ct]

        if ct in n.config["synfuel_import_force"]:
            synfuel_imports = n.config["synfuel_import_force"][ct].get(investment_year,0)*1e6
            logger.info(f"Subtracting synfuel imports of {synfuel_imports} from {ct} CO2 target")
            limit += 0.27*synfuel_imports

        logger.info(f"Limiting emissions in country {ct} to {limit_countries[ct]} of 1990 levels, i.e. {limit} tCO2/a")

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:

            links = n.links.index[(n.links.index.str[:2] == ct) & (n.links[f"bus{port}"] == "co2 atmosphere")]

            logger.info(f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links,'carrier'].unique()}")

            if port == "0":
                efficiency = -1.
            elif port == "1":
                efficiency = n.links.loc[links, f"efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append((n.model["Link-p"].loc[:, links]*efficiency*n.snapshot_weightings.generators).sum())

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )
        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )

def force_boiler_profiles_existing_per_load(n):
    """this scales the boiler dispatch to the load profile with a factor common to all boilers at load"""

    logger.info("Forcing boiler profiles for existing ones")

    decentral_boilers = n.links.index[n.links.carrier.str.contains("boiler")
                                      & ~n.links.carrier.str.contains("urban central")
                                      & ~n.links.p_nom_extendable]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers,"bus1"]
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(n.loads_t.p_set[boiler_loads].max(),axis=1)
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(boiler_profiles_pu.multiply(n.links.loc[decentral_boilers,"p_nom"],axis=1))

    boiler_load_index = pd.Index(boiler_loads.unique())
    boiler_load_index.name = "Load"

    # per load scaling factor
    n.model.add_variables(coords=[boiler_load_index], name="Load-profile_factor")

    # clumsy indicator matrix to map boilers to loads
    df = pd.DataFrame(index=boiler_load_index,columns=decentral_boilers,data=0.)
    for k,v in boiler_loads.items():
        df.loc[v,k] = 1.

    lhs = n.model["Link-p"].loc[:,decentral_boilers] - (boiler_profiles*DataArray(df)*n.model["Load-profile_factor"]).sum("Load")

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.loads["profile_factor_opt"] = 0.


def force_boiler_profiles_existing_per_boiler(n):
    """this scales each boiler dispatch to be proportional to the load profile"""

    logger.info("Forcing each existing boiler dispatch to be proportional to the load profile")

    decentral_boilers = n.links.index[n.links.carrier.str.contains("boiler")
                                      & ~n.links.carrier.str.contains("urban central")
                                      & ~n.links.p_nom_extendable]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers,"bus1"]
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(n.loads_t.p_set[boiler_loads].max(),axis=1)
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(boiler_profiles_pu.multiply(n.links.loc[decentral_boilers,"p_nom"],axis=1))

    #will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (1, n.model["Link-p"].loc[:,decentral_boilers]), (-boiler_profiles, n.model["Link-fixed_profile_scaling"])

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.


def force_synfuel_import_demand(n, snakemake, investment_year):

    for ct in n.config["synfuel_import_force"]:
        synfuel_imports = n.config["synfuel_import_force"][ct].get(investment_year,0)*1e6
        logger.info(f"Accounting for synfuel imports of {synfuel_imports} in {ct}")


        demand = n.links.index[(n.links.bus0 == "EU oil") & (n.links.index.str[:2] == ct)]
        local_synproduction = n.links.index[(n.links.bus1 == "EU oil") & (n.links.index.str[:2] == ct)]

        demand_p = (n.model["Link-p"].loc[:, demand]*n.snapshot_weightings.generators).sum()
        local_synproduction_p = (n.model["Link-p"].loc[:, local_synproduction]*n.snapshot_weightings.generators).sum()

        lhs = demand_p - local_synproduction_p

        cname = f"force_synfuel_import-{ct}"

        n.model.add_constraints(
            lhs >= synfuel_imports, name=f"GlobalConstraint-{cname}"
        )
        n.add(
            "GlobalConstraint",
            cname,
            constant=synfuel_imports,
            sense=">=",
            type="",
            carrier_attribute="",
        )



def additional_functionality(n, snapshots, snakemake):

    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])

    add_min_limits(n, snapshots, investment_year, snakemake.config)

    h2_import_limits(n, snapshots, investment_year, snakemake.config)

    #force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    force_synfuel_import_demand(n, snakemake, investment_year)

    if snakemake.config["sector"]["co2_budget_national"]:
        limit_countries = snakemake.config["co2_budget_national"][investment_year]
        add_co2limit_country(n, limit_countries, snakemake, investment_year)
