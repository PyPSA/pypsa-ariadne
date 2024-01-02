
import logging, pandas as pd

from prepare_sector_network import emission_sectors_from_opts

logger = logging.getLogger(__name__)


def add_min_limits(n, snapshots, investment_year, config):

    logger.info(config["limits_min"])

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

                n.model.add_constraints(
                    lhs >= limit - existing_capacity, name=f"GlobalConstraint-{c.name}-{carrier.replace(' ','-')}-capacity"
                )

def h2_import_limits(n, snapshots, investment_year, config):

    for ct in config["h2_import_max"]:
        limit = config["h2_import_max"][ct][investment_year]*1e6

        logger.info(f"limiting H2 imports in {ct} to {limit/1e6} TWh/a")

        incoming = n.links.index[(n.links.carrier == "H2 pipeline") & (n.links.bus0.str[:2] != ct) & (n.links.bus1.str[:2] == ct)]
        outgoing = n.links.index[(n.links.carrier == "H2 pipeline") & (n.links.bus0.str[:2] == ct) & (n.links.bus1.str[:2] != ct)]

        incoming_p = (n.model["Link-p"].loc[:, incoming]*n.snapshot_weightings.generators).sum()
        outgoing_p = (n.model["Link-p"].loc[:, outgoing]*n.snapshot_weightings.generators).sum()

        logger.info(incoming_p)
        logger.info(outgoing_p)

        lhs = incoming_p - outgoing_p

        logger.info(lhs)

        n.model.add_constraints(
            lhs <= limit, name=f"GlobalConstraint-H2_import_limit-{ct}"
        )


def add_co2limit_country(n, limit_countries, snakemake, nyears=1.0):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_country_{investment_year}'.

    Parameters
    ----------
    n : pypsa.Network
    config : dict
    limit_countries : dict
    nyears: float, optional
        Used to scale the emissions constraint to the number of snapshots of the base network.
    """
    logger.info(f"Adding CO2 budget limit for each country as per unit of 1990 levels")

    countries = n.config["countries"]

    # TODO: import function from prepare_sector_network? Move to common place?
    sectors = emission_sectors_from_opts(n.opts)

    # convert Mt to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_limit_countries = co2_totals.loc[countries, sectors].sum(axis=1)
    co2_limit_countries = co2_limit_countries.loc[co2_limit_countries.index.isin(limit_countries.keys())]

    co2_limit_countries *= co2_limit_countries.index.map(limit_countries) * nyears

    p = n.model["Link-p"]  # dimension: (time, component)

    # NB: Most country-specific links retain their locational information in bus1 (except for DAC, where it is in bus2, and process emissions, where it is in bus0)
    country = n.links.bus1.map(n.buses.location).map(n.buses.country)
    country_DAC = (
        n.links[n.links.carrier == "DAC"]
        .bus2.map(n.buses.location)
        .map(n.buses.country)
    )
    country[country_DAC.index] = country_DAC
    country_process_emissions = (
        n.links[n.links.carrier.str.contains("process emissions")]
        .bus0.map(n.buses.location)
        .map(n.buses.country)
    )
    country[country_process_emissions.index] = country_process_emissions

    lhs = []
    for port in [col[3:] for col in n.links if col.startswith("bus")]:
        if port == str(0):
            efficiency = (
                n.links["efficiency"].apply(lambda x: -1.0).rename("efficiency0")
            )
        elif port == str(1):
            efficiency = n.links["efficiency"]
        else:
            efficiency = n.links[f"efficiency{port}"]
        mask = n.links[f"bus{port}"].map(n.buses.carrier).eq("co2")

        idx = n.links[mask].index

        international = n.links.carrier.map(
            lambda x: 0.4 if x in ["kerosene for aviation", "shipping oil"] else 1.0
        )
        grouping = country.loc[idx]

        if not grouping.isnull().all():
            expr = (
                ((p.loc[:, idx] * efficiency[idx] * international[idx])
                .groupby(grouping, axis=1)
                .sum()
                *n.snapshot_weightings.generators
                )
                .sum(dims="snapshot")
            )
            lhs.append(expr)

    lhs = sum(lhs)  # dimension: (country)
    lhs = lhs.rename({list(lhs.dims.keys())[0]: "snapshot"})
    rhs = pd.Series(co2_limit_countries)  # dimension: (country)

    for ct in lhs.indexes["snapshot"]:
        n.model.add_constraints(
            lhs.loc[ct] <= rhs[ct],
            name=f"GlobalConstraint-co2_limit_per_country{ct}",
        )
        n.add(
            "GlobalConstraint",
            f"co2_limit_per_country{ct}",
            constant=rhs[ct],
            sense="<=",
            type="",
        )


def additional_functionality(n, snapshots, snakemake):

    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])

    add_min_limits(n, snapshots, investment_year, snakemake.config)

    h2_import_limits(n, snapshots, investment_year, snakemake.config)

    if snakemake.config["sector"]["co2_budget_national"]:
        # prepare co2 constraint
        nhours = n.snapshot_weightings.generators.sum()
        nyears = nhours / 8760
        limit_countries = snakemake.config["co2_budget_national"][investment_year]

        # add co2 constraint for each country
        logger.info(f"Add CO2 limit for each country")
        add_co2limit_country(n, limit_countries, snakemake, nyears)
