# -*- coding: utf-8 -*-

import logging

import pandas as pd
from prepare_sector_network import determine_emission_sectors
from xarray import DataArray

logger = logging.getLogger(__name__)


def add_capacity_limits(n, investment_year, limits_capacity, sense="maximum"):

    for c in n.iterate_components(limits_capacity):
        logger.info(f"Adding {sense} constraints for {c.list_name}")

        attr = "e" if c.name == "Store" else "p"
        units = "MWh or tCO2" if c.name == "Store" else "MW"

        for carrier in limits_capacity[c.name]:

            for ct in limits_capacity[c.name][carrier]:
                if investment_year not in limits_capacity[c.name][carrier][ct].keys():
                    continue

                limit = 1e3 * limits_capacity[c.name][carrier][ct][investment_year]

                logger.info(
                    f"Adding constraint on {c.name} {carrier} capacity in {ct} to be {sense} {limit} {units}"
                )

                valid_components = (
                    (c.df.index.str[:2] == ct)
                    & (c.df.carrier.str[: len(carrier)] == carrier)
                    & ~c.df.carrier.str.contains("thermal")
                )  # exclude solar thermal

                existing_index = c.df.index[
                    valid_components & ~c.df[attr + "_nom_extendable"]
                ]
                extendable_index = c.df.index[
                    valid_components & c.df[attr + "_nom_extendable"]
                ]

                existing_capacity = c.df.loc[existing_index, attr + "_nom"].sum()

                logger.info(
                    f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} {units}"
                )

                nom = n.model[c.name + "-" + attr + "_nom"].loc[extendable_index]

                lhs = nom.sum()

                cname = f"capacity_{sense}-{ct}-{c.name}-{carrier.replace(' ','-')}"

                if cname in n.global_constraints.index:
                    logger.warning(
                        f"Global constraint {cname} already exists. Dropping and adding it again."
                    )
                    n.global_constraints.drop(cname, inplace=True)

                rhs = limit - existing_capacity

                if sense == "maximum":
                    if rhs <= 0:
                        logger.warning(
                            f"Existing capacity in {ct} for carrier {carrier} already exceeds the limit of {limit} MW. Limiting capacity expansion for this investment period to 0."
                        )
                        rhs = 0

                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense="<=",
                        type="",
                        carrier_attribute="",
                    )

                elif sense == "minimum":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense=">=",
                        type="",
                        carrier_attribute="",
                    )
                else:
                    logger.error("sense {sense} not recognised")
                    sys.exit()


def add_power_limits(n, investment_year, limits_power_max):
    """
    " Restricts the maximum inflow/outflow of electricity from/to a country.
    """
    for ct in limits_power_max:
        if investment_year not in limits_power_max[ct].keys():
            continue

        limit = 1e3 * limits_power_max[ct][investment_year] / 10

        logger.info(
            f"Adding constraint on electricity import/export from/to {ct} to be < {limit} MW"
        )
        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        # iterate over snapshots - otherwise exporting of postnetwork fails since
        # the constraints are time dependent
        for t in n.snapshots:
            incoming_line_p = n.model["Line-s"].loc[t, incoming_line]
            outgoing_line_p = n.model["Line-s"].loc[t, outgoing_line]
            incoming_link_p = n.model["Link-p"].loc[t, incoming_link]
            outgoing_link_p = n.model["Link-p"].loc[t, outgoing_link]

            lhs = (
                incoming_link_p.sum()
                - outgoing_link_p.sum()
                + incoming_line_p.sum()
                - outgoing_line_p.sum()
            ) / 10
            # divide by 10 to avoid numerical issues

            cname_upper = f"Power-import-limit-{ct}-{t}"
            cname_lower = f"Power-export-limit-{ct}-{t}"

            n.model.add_constraints(lhs <= limit, name=cname_upper)
            n.model.add_constraints(lhs >= -limit, name=cname_lower)

            # not adding to network as the shadow prices are not needed


def h2_import_limits(n, investment_year, limits_volume_max):

    for ct in limits_volume_max["h2_import"]:
        if investment_year not in limits_volume_max["h2_import"][ct].keys():
            continue
        limit = limits_volume_max["h2_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 imports in {ct} to {limit/1e6} TWh/a")
        pipeline_carrier = [
            "H2 pipeline",
            "H2 pipeline (Kernnetz)",
            "H2 pipeline retrofitted",
        ]
        incoming = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_p = (
            n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
        ).sum()
        outgoing_p = (
            n.model["Link-p"].loc[:, outgoing] * n.snapshot_weightings.generators
        ).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def h2_production_limits(n, investment_year, limits_volume_min, limits_volume_max):

    for ct in limits_volume_max["electrolysis"]:
        if ct not in limits_volume_min["electrolysis"]:
            logger.warning(
                f"no lower limit for H2 electrolysis in {ct} assuming 0 TWh/a"
            )
            limit_lower = 0
        else:
            limit_lower = limits_volume_min["electrolysis"][ct][investment_year] * 1e6

        limit_upper = limits_volume_max["electrolysis"][ct][investment_year] * 1e6

        logger.info(
            f"limiting H2 electrolysis in DE between {limit_lower/1e6} and {limit_upper/1e6} TWh/a"
        )

        production = n.links[
            (n.links.carrier == "H2 Electrolysis") & (n.links.bus0.str.contains(ct))
        ].index

        lhs = (
            n.model["Link-p"].loc[:, production] * n.snapshot_weightings.generators
        ).sum()

        cname_upper = f"H2_production_limit_upper-{ct}"
        cname_lower = f"H2_production_limit_lower-{ct}"

        n.model.add_constraints(
            lhs <= limit_upper, name=f"GlobalConstraint-{cname_upper}"
        )

        n.model.add_constraints(
            lhs >= limit_lower, name=f"GlobalConstraint-{cname_lower}"
        )

        if cname_upper not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_upper,
                constant=limit_upper,
                sense="<=",
                type="",
                carrier_attribute="",
            )
        if cname_lower not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_lower,
                constant=limit_lower,
                sense=">=",
                type="",
                carrier_attribute="",
            )


def electricity_import_limits(n, investment_year, limits_volume_max):

    for ct in limits_volume_max["electricity_import"]:
        limit = limits_volume_max["electricity_import"][ct][investment_year] * 1e6

        logger.info(f"limiting electricity imports in {ct} to {limit/1e6} TWh/a")

        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_line_p = (
            n.model["Line-s"].loc[:, incoming_line] * n.snapshot_weightings.generators
        ).sum()
        outgoing_line_p = (
            n.model["Line-s"].loc[:, outgoing_line] * n.snapshot_weightings.generators
        ).sum()

        incoming_link_p = (
            n.model["Link-p"].loc[:, incoming_link] * n.snapshot_weightings.generators
        ).sum()
        outgoing_link_p = (
            n.model["Link-p"].loc[:, outgoing_link] * n.snapshot_weightings.generators
        ).sum()

        lhs = (incoming_link_p - outgoing_link_p) + (incoming_line_p - outgoing_line_p)

        cname = f"Electricity_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def add_co2limit_country(n, limit_countries, snakemake, debug=False):
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

    sectors = determine_emission_sectors(n.config["sector"])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in limit_countries:
        limit = co2_total_totals[ct] * limit_countries[ct]
        logger.info(
            f"Limiting emissions in country {ct} to {limit_countries[ct]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:

            links = n.links.index[
                (n.links.index.str[:2] == ct)
                & (n.links[f"bus{port}"] == "co2 atmosphere")
            ]

            logger.info(
                f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links,'carrier'].unique()}"
            )

            if port == "0":
                efficiency = -1.0
            elif port == "1":
                efficiency = n.links.loc[links, f"efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append(
                (
                    n.model["Link-p"].loc[:, links]
                    * efficiency
                    * n.snapshot_weightings.generators
                ).sum()
            )

        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == "EU renewable oil -> DE oil"]
        outgoing_oil = n.links.index[n.links.index == "DE renewable oil -> EU oil"]

        if not debug:
            lhs.append(
                (
                    -1
                    * n.model["Link-p"].loc[:, incoming_oil]
                    * 0.2571
                    * n.snapshot_weightings.generators
                ).sum()
            )
            lhs.append(
                (
                    n.model["Link-p"].loc[:, outgoing_oil]
                    * 0.2571
                    * n.snapshot_weightings.generators
                ).sum()
            )

        incoming_methanol = n.links.index[n.links.index == "EU methanol -> DE methanol"]
        outgoing_methanol = n.links.index[n.links.index == "DE methanol -> EU methanol"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        # Methane
        incoming_CH4 = n.links.index[n.links.index == "EU renewable gas -> DE gas"]
        outgoing_CH4 = n.links.index[n.links.index == "DE renewable gas -> EU gas"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def force_boiler_profiles_existing_per_load(n):
    """
    This scales the boiler dispatch to the load profile with a factor common to
    all boilers at load.
    """

    logger.info("Forcing boiler profiles for existing ones")

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    boiler_load_index = pd.Index(boiler_loads.unique())
    boiler_load_index.name = "Load"

    # per load scaling factor
    n.model.add_variables(coords=[boiler_load_index], name="Load-profile_factor")

    # clumsy indicator matrix to map boilers to loads
    df = pd.DataFrame(index=boiler_load_index, columns=decentral_boilers, data=0.0)
    for k, v in boiler_loads.items():
        df.loc[v, k] = 1.0

    lhs = n.model["Link-p"].loc[:, decentral_boilers] - (
        boiler_profiles * DataArray(df) * n.model["Load-profile_factor"]
    ).sum("Load")

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.loads["profile_factor_opt"] = 0.0


def force_boiler_profiles_existing_per_boiler(n):
    """
    This scales each boiler dispatch to be proportional to the load profile.
    """

    logger.info(
        "Forcing each existing boiler dispatch to be proportional to the load profile"
    )

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    # will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (1, n.model["Link-p"].loc[:, decentral_boilers]), (
        -boiler_profiles,
        n.model["Link-fixed_profile_scaling"],
    )

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.0


def add_h2_derivate_limit(n, investment_year, limits_volume_max, limit_non_eu_de):

    for ct in limits_volume_max["h2_derivate_import"]:
        if investment_year not in limits_volume_max["h2_derivate_import"][ct].keys():
            continue
        limit = limits_volume_max["h2_derivate_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 derivate imports in {ct} to {limit/1e6} TWh/a")

        incoming = n.links.loc[
            [
                "EU renewable oil -> DE oil",
                "EU methanol -> DE methanol",
                "EU renewable gas -> DE gas",
            ]
        ].index
        if not limit_non_eu_de:
            outgoing = n.links.loc[
                [
                    "DE renewable oil -> EU oil",
                    "DE methanol -> EU methanol",
                    "DE renewable gas -> EU gas",
                ]
            ].index
            outgoing_p = (
                n.model["Link-p"].loc[:, outgoing] * n.snapshot_weightings.generators
            ).sum()
        else:
            outgoing_p = 0

        incoming_p = (
            n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
        ).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_derivate_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def adapt_nuclear_output(n):

    logger.info(
        f"limiting german electricity generation from nuclear to 2020 value of 61 TWh"
    )
    limit = 61e6

    nuclear_de_index = n.links.index[
        (n.links.carrier == "nuclear") & (n.links.index.str[:2] == "DE")
    ]

    nuclear_gen = (
        n.model["Link-p"].loc[:, nuclear_de_index]
        * n.links.loc[nuclear_de_index, "efficiency"]
        * n.snapshot_weightings.generators
    ).sum()

    lhs = nuclear_gen

    cname = f"Nuclear_generation_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if cname in n.global_constraints.index:
        logger.warning(
            f"Global constraint {cname} already exists. Dropping and adding it again."
        )
        n.global_constraints.drop(cname, inplace=True)

    n.add(
        "GlobalConstraint",
        cname,
        constant=limit,
        sense="<=",
        type="",
        carrier_attribute="",
    )


def FT_production_limit(n, investment_year, config):
    """ "
    Limit the production of FT fuels in a country to a certain volume.
    """

    for ct in config["FT_production"]:
        limit = config["FT_production"][ct][investment_year] * 1e6

        logger.info(f"limiting FT production in {ct} to {limit/1e6} TWh/a")

        prod_links = n.links[
            (n.links.index.str[:2] == "DE") & (n.links.carrier == "Fischer-Tropsch")
        ].index

        prod_volume = (
            n.model["Link-p"].loc[:, prod_links] * n.snapshot_weightings.generators
        ).sum() / 100
        # avoid large bounds
        limit /= 100

        cname = f"FT_production_volume_limit-{ct}"

        n.model.add_constraints(prod_volume <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def remove_production_limits(n, investment_year, limits_volume_max):
    """"
    Remove any restrictions on the production volumes of FT and H2.
    Adding steel, hbi and ammonia to the the import restrictions for H2 derivatives.
    """

    logger.info("Removing any EU production limit constraints.")

    cnames = [
        "H2_derivate_import_limit-DE",
        "FT_production_volume_limit-DE",
        "H2_production_limit_upper-DE",
        "H2_production_limit_lower-DE",
        # "H2_import_limit-DE",
        # "Electricity_import_limit-DE",
    ]

    for cname in cnames:
        if cname in n.global_constraints.index:
            logger.info(f"Removing constraint {cname}")
            n.global_constraints.drop(cname, inplace=True)
            n.model.constraints.remove("GlobalConstraint-" + cname)

    if investment_year not in limits_volume_max["h2_derivate_import"]["DE"].keys():
            return
    logger.info("Adding European H2 derivative import limit.")

    limit = limits_volume_max["h2_derivate_import"]["DE"][investment_year] * 1e6

    logger.info(f"limiting H2 derivate imports in DE to {limit/1e6} TWh/a")

    incoming = n.links.loc[
        [
            "EU renewable oil -> DE oil",
            "EU methanol -> DE methanol",
            "EU renewable gas -> DE gas",
            "EU NH3 -> DE NH3",
            "EU steel -> DE steel",
            "EU hbi -> DE hbi",
        ]
    ].index

    lhs = (
        n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
    ).sum()

    cname = "H2_derivate_import_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if cname in n.global_constraints.index:
        logger.warning(
            f"Global constraint {cname} already exists. Dropping and adding it again."
        )
        n.global_constraints.drop(cname, inplace=True)

    n.add(
        "GlobalConstraint",
        cname,
        constant=limit,
        sense="<=",
        type="",
        carrier_attribute="",
    )


def import_limit_eu(n, sns, limit_eu_de, investment_year):
    """
    Limiting European imports to Germany to net 0 TWh for each carrier.
    """

    rhs = 0

    logger.info("Limiting European imports to Germany to net 0 TWh for each carrier.")
    ct = "DE"
    # collect indices of more complex carriers
    h2_in = n.links.index[(n.links.carrier.str.contains("H2 pipeline")) & 
                        (n.links.bus0.str[:2] != ct) &
                        (n.links.bus1.str[:2] == ct)]
    h2_out = n.links.index[(n.links.carrier.str.contains("H2 pipeline")) & 
                        (n.links.bus0.str[:2] == ct) &
                        (n.links.bus1.str[:2] != ct)]
    elec_links_in = n.links.index[((n.links.carrier == "DC") | (n.links.carrier == "AC")) & (n.links.bus0.str[:2] != ct) & (n.links.bus1.str[:2] == ct)]
    elec_links_out = n.links.index[((n.links.carrier == "DC") | (n.links.carrier == "AC")) & (n.links.bus0.str[:2] == ct) & (n.links.bus1.str[:2] != ct)]

    elec_lines_in = n.lines.index[(n.lines.carrier == "AC") & (n.lines.bus0.str[:2] != ct) & (n.lines.bus1.str[:2] == ct)]
    elec_lines_out = n.lines.index[(n.lines.carrier == "AC") & (n.lines.bus0.str[:2] == ct) & (n.lines.bus1.str[:2] != ct)]
    
    lhs_ele_s = n.model["Line-s"].loc[sns, elec_lines_in].sum() - n.model["Line-s"].loc[sns, elec_lines_out].sum()
    lhs_ele_p = n.model["Link-p"].loc[sns, elec_links_in].sum() - n.model["Link-p"].loc[sns, elec_links_out].sum()
    
    lhs_h2 = n.model["Link-p"].loc[sns, h2_in].sum() - n.model["Link-p"].loc[sns, h2_out].sum()
    
    lhs_ft = n.model["Link-p"].loc[sns, "EU renewable oil -> DE oil"].sum() - n.model["Link-p"].loc[sns, "DE renewable oil -> EU oil"].sum()
    lhs_meoh = n.model["Link-p"].loc[sns, "EU methanol -> DE methanol"].sum() - n.model["Link-p"].loc[sns, "DE methanol -> EU methanol"].sum()
    lhs_gas = n.model["Link-p"].loc[sns, "EU renewable gas -> DE gas"].sum() - n.model["Link-p"].loc[sns, "DE renewable gas -> EU gas"].sum()

    lhs_nh3 = n.model["Link-p"].loc[sns, "EU NH3 -> DE NH3"].sum() - n.model["Link-p"].loc[sns, "DE NH3 -> EU NH3"].sum()

    lhs_steel = n.model["Link-p"].loc[sns, "EU steel -> DE steel"].sum() - n.model["Link-p"].loc[sns, "DE steel -> EU steel"].sum()
    lhs_hbi = n.model["Link-p"].loc[sns, "EU hbi -> DE hbi"].sum() - n.model["Link-p"].loc[sns, "DE hbi -> EU hbi"].sum()
    # electricity
    n.model.add_constraints(lhs_ele_s, "==", rhs, name="import_limit_lines")
    n.model.add_constraints(lhs_ele_p, "==", rhs, name="import_limit_ele")
    # hydrogen
    n.model.add_constraints(lhs_h2, "==", rhs, name="import_limit_h2")
    # h2 derivatives
    n.model.add_constraints(lhs_ft, "==", rhs, name="import_limit_ft")
    n.model.add_constraints(lhs_meoh, "==", rhs, name="import_limit_meoh")
    n.model.add_constraints(lhs_gas, "==", rhs, name="import_limit_gas")
    n.model.add_constraints(lhs_nh3, "==", rhs, name="import_limit_nh3")
    # steel and hbi
    n.model.add_constraints(lhs_steel, "==", rhs, name="import_limit_steel")
    n.model.add_constraints(lhs_hbi, "==", rhs, name="import_limit_hbi")


def import_limit_non_eu(n, sns, limit_non_eu_de, investment_year):
    
    logger.info("Adding a limit of {limit_non_eu_de} TWh of non-European imports.")

    # get all non-European import links
    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE") &
        (n.links.carrier.str.contains("import"))
        ].index

    if non_eu_links.empty:
        logger.warning("No non-European import links found but limit_non_eu_de is set. Please check config[solving][constraints][limit_non_eu_de] and config[import][enable].")
        return

    weightings = n.snapshot_weightings.loc[sns, "generators"]

    p_links = n.model["Link-p"].loc[sns, non_eu_links]

    lhs = (p_links * weightings).sum()

    rhs = limit_non_eu_de * 1e6

    n.model.add_constraints(lhs, "==", rhs, name="energy_import_limit")

    # restrict hydrogen export
    h2_links = n.links.index[
        (n.links.bus0.str[:2] == "DE") &
        (n.links.bus1.str[:2] != "DE") &
        (n.links.carrier.str.contains("H2 pipeline"))
    ]
    lhs = (
            n.model["Link-p"].loc[sns, h2_links] * n.snapshot_weightings.generators
        ).sum()

    n.model.add_constraints(lhs, "<=", 0, name="h2_export_limit_DE")
    
    # might be necessary to add constraint for electricity export as well


def FT_production_limit(n, investment_year, config):
    """ "
    Limit the production of FT fuels in a country to a certain volume.
    """

    for ct in config["FT_production"]:
        limit = config["FT_production"][ct][investment_year] * 1e6

        logger.info(f"limiting FT production in {ct} to {limit/1e6} TWh/a")

        prod_links = n.links[
            (n.links.index.str[:2] == "DE") & (n.links.carrier == "Fischer-Tropsch")
        ].index

        prod_volume = (
            n.model["Link-p"].loc[:, prod_links] * n.snapshot_weightings.generators
        ).sum() / 100
        # avoid large bounds
        limit /= 100

        cname = f"FT_production_volume_limit-{ct}"

        n.model.add_constraints(prod_volume <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def ramp_up_limit_non_EU(n, n_snapshots, limits_volume_max, investment_year):

    if investment_year not in limits_volume_max["h2_derivate_import"]["DE"].keys():
        return
    limit = limits_volume_max["h2_derivate_import"]["DE"][investment_year] * 1e6

    logger.info(f"limiting non European H2 derivate imports to DE to {limit/1e6} TWh/a")

    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE") &
        (n.links.carrier.str.contains("import")) &
        ~(n.links.carrier.str.contains("h2"))
        ].index

    incoming_p = (
        n.model["Link-p"].loc[:, non_eu_links] * n.snapshot_weightings.generators
    ).sum()

    lhs = incoming_p

    cname = "non_European_H2_derivate_import_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if investment_year not in limits_volume_max["h2_import"]["DE"].keys():
        return
    limit = limits_volume_max["h2_import"]["DE"][investment_year] * 1e6

    logger.info(f"limiting non European H2 imports to DE to {limit/1e6} TWh/a")

    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE") &
        (n.links.carrier.str.contains("import")) &
        (n.links.carrier.str.contains("h2"))
        ].index

    incoming_p = (
        n.model["Link-p"].loc[:, non_eu_links] * n.snapshot_weightings.generators
    ).sum()

    lhs = incoming_p

    cname = "non_European_H2_import_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")



def additional_functionality(n, snapshots, snakemake):

    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    constraints = snakemake.params.solving["constraints"]

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_min"], "minimum"
    )

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_max"], "maximum"
    )

    add_power_limits(n, investment_year, constraints["limits_power_max"])

    FT_production_limit(n, investment_year, constraints["limits_volume_max"])

    if int(snakemake.wildcards.clusters) != 1:
        h2_import_limits(n, investment_year, constraints["limits_volume_max"])

        electricity_import_limits(n, investment_year, constraints["limits_volume_max"])

    if investment_year >= 2025:
        h2_production_limits(
            n,
            investment_year,
            constraints["limits_volume_min"],
            constraints["limits_volume_max"],
        )

    if not snakemake.config["run"]["debug_h2deriv_limit"]:
        add_h2_derivate_limit(n, investment_year, constraints["limits_volume_max"], constraints["limit_non_eu_de"])

    # force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    if isinstance(constraints["co2_budget_national"], dict):
        limit_countries = constraints["co2_budget_national"][investment_year]
        add_co2limit_country(
            n,
            limit_countries,
            snakemake,
            debug=snakemake.config["run"]["debug_co2_limit"],
        )
    else:
        logger.warning("No national CO2 budget specified!")

    if investment_year == 2020:
        adapt_nuclear_output(n)

    limit_eu_de = constraints["limit_eu_de"]
    limit_non_eu_de = constraints["limit_non_eu_de"]

    # Remove any production volume constraints - change derivative import constraint
    remove_production_limits(n, investment_year, constraints["limits_volume_max"])

    if limit_eu_de:
        logger.info("Adding import limit for European imports to Germany.")
        import_limit_eu(n, snapshots, limit_eu_de, investment_year)
    if 'import shipping-lh2' in n.links.carrier.unique():
        logger.info("Ramp up import limit for non European imports to Germany.")
        ramp_up_limit_non_EU(n, snapshots, constraints["limits_volume_max"], investment_year)
    if limit_non_eu_de:
        logger.info("Adding import limit for non European imports to Germany.")
        import_limit_non_eu(n, snapshots, limit_non_eu_de, investment_year)