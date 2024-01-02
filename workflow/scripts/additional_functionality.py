
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


def add_co2limit_country(n, limit_countries, snakemake):
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

    sectors = emission_sectors_from_opts(n.opts)

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in limit_countries:
        limit = co2_total_totals[ct]*limit_countries[ct]
        logger.info(f"Limiting emissions in country {ct} to {limit_countries[ct]} of 1990 levels, i.e. {limit} tCO2/a")

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:

            links = n.links.index[(n.links.index.str[:2] == ct) & (n.links[f"bus{port}"] == "co2 atmosphere")]

            if port == "0":
                efficiency = -1.
            elif port == "1":
                efficiency = n.links.loc[links, f"efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            international_factor = pd.Series(1., index=links)
            # TODO: move to config
            international_factor[links.str.contains("shipping oil")] = 0.4
            international_factor[links.str.contains("kerosene for aviation")] = 0.4

            lhs.append((n.model["Link-p"].loc[:, links]*efficiency*international_factor*n.snapshot_weightings.generators).sum())

        lhs = sum(lhs)

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-co2_limit-{ct}",
        )
        n.add(
            "GlobalConstraint",
            f"co2_limit-{ct}",
            constant=limit,
            sense="<=",
            type="",
        )


def additional_functionality(n, snapshots, snakemake):

    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])

    add_min_limits(n, snapshots, investment_year, snakemake.config)

    h2_import_limits(n, snapshots, investment_year, snakemake.config)

    if snakemake.config["sector"]["co2_budget_national"]:
        logger.info(f"Add CO2 limit for each country")

        limit_countries = snakemake.config["co2_budget_national"][investment_year]

        add_co2limit_country(n, limit_countries, snakemake)
