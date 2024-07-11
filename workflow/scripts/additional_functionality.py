
import logging

import pandas as pd
from prepare_sector_network import determine_emission_sectors

from xarray import DataArray

logger = logging.getLogger(__name__)


def add_min_limits(n, investment_year, config):

    for c in n.iterate_components(config["limits_capacity_min"]):
        logger.info(f"Adding minimum constraints for {c.list_name}")

        for carrier in config["limits_capacity_min"][c.name]:

            for ct in config["limits_capacity_min"][c.name][carrier]:
                # check if the limit is defined for the investement year
                if investment_year not in config["limits_capacity_min"][c.name][carrier][ct].keys():
                    continue
                limit = 1e3*config["limits_capacity_min"][c.name][carrier][ct][investment_year]

                logger.info(f"Adding constraint on {c.name} {carrier} capacity in {ct} to be greater than {limit} MW")

                valid_components = (
                    (c.df.index.str[:2] == ct) &
                    (c.df.carrier.str[:len(carrier)] == carrier) &
                    ~c.df.carrier.str.contains("thermal")) # exclude solar thermal
                
                existing_index = c.df.index[valid_components & ~c.df.p_nom_extendable]
                extendable_index = c.df.index[valid_components & c.df.p_nom_extendable]

                existing_capacity = c.df.loc[existing_index, "p_nom"].sum()

                logger.info(f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} MW")

                p_nom = n.model[c.name + "-p_nom"].loc[extendable_index]

                lhs = p_nom.sum()

                cname = f"capacity_minimum-{ct}-{c.name}-{carrier.replace(' ','-')}"

                n.model.add_constraints(
                    lhs >= limit - existing_capacity, name=f"GlobalConstraint-{cname}"
                )
                if cname not in n.global_constraints.index:
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=limit,
                        sense=">=",
                        type="",
                        carrier_attribute="",
                    )
                
def add_max_limits(n, investment_year, config):

    for c in n.iterate_components(config["limits_capacity_max"]):
        logger.info(f"Adding maximum constraints for {c.list_name}")

        for carrier in config["limits_capacity_max"][c.name]:

            for ct in config["limits_capacity_max"][c.name][carrier]:
                if investment_year not in config["limits_capacity_max"][c.name][carrier][ct].keys():
                    continue
                limit = 1e3*config["limits_capacity_max"][c.name][carrier][ct][investment_year]

                valid_components = (
                    (c.df.index.str[:2] == ct) &
                    (c.df.carrier.str[:len(carrier)] == carrier) &
                    ~c.df.carrier.str.contains("thermal")) # exclude solar thermal
                
                existing_index = c.df.index[valid_components & ~c.df.p_nom_extendable]
                extendable_index = c.df.index[valid_components & c.df.p_nom_extendable]

                existing_capacity = c.df.loc[existing_index, "p_nom"].sum()

                logger.info(f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} MW")
                logger.info(f"Adding constraint on {c.name} {carrier} capacity in {ct} to be smaller than {limit} MW")

                p_nom = n.model[c.name + "-p_nom"].loc[extendable_index]

                lhs = p_nom.sum()

                cname = f"capacity_maximum-{ct}-{c.name}-{carrier.replace(' ','-')}"
                if limit - existing_capacity <= 0:
                    n.model.add_constraints(
                        lhs <= 0, name=f"GlobalConstraint-{cname}"
                    )
                    logger.warning(f"Existing capacity in {ct} for carrier {carrier} already exceeds the limit of {limit} MW. Limiting capacity expansion for this investment period to 0.")
                else:
                    n.model.add_constraints(
                        lhs <= limit - existing_capacity, name=f"GlobalConstraint-{cname}"
                    )
                if cname not in n.global_constraints.index:
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=limit,
                        sense="<=",
                        type="",
                        carrier_attribute="",
                    )


def h2_import_limits(n, snapshots, investment_year, config):

    for ct in config["limits_volume_max"]["h2_import"]:
        limit = config["limits_volume_max"]["h2_import"][ct][investment_year]*1e6

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
        
        if cname not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname,
                constant=limit,
                sense="<=",
                type="",
                carrier_attribute="",
            )

def h2_production_limits(n, snapshots, investment_year, config):

    for ct in config["limits_volume_max"]["electrolysis"]:
        if ct not in config["limits_volume_min"]["electrolysis"]:
            logger.warning(f"no lower limit for H2 electrolysis in {ct} assuming 0 TWh/a")
            limit_lower = 0
        else:
            limit_lower = config["limits_volume_min"]["electrolysis"][ct][investment_year]*1e6
        
        limit_upper = config["limits_volume_max"]["electrolysis"][ct][investment_year]*1e6

        logger.info(f"limiting H2 electrolysis in DE between {limit_lower/1e6} and {limit_upper/1e6} TWh/a")

        production = n.links[(n.links.carrier == "H2 Electrolysis") & (n.links.bus0.str.contains(ct))].index

        lhs = (n.model["Link-p"].loc[:, production]*n.snapshot_weightings.generators).sum()

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


def electricity_import_limits(n, snapshots, investment_year, config):

    for ct in config["limits_volume_max"]["electricity_import"]:
        limit = config["limits_volume_max"]["electricity_import"][ct][investment_year]*1e6

        logger.info(f"limiting electricity imports in {ct} to {limit/1e6} TWh/a")

        incoming_line = n.lines.index[(n.lines.carrier == "AC") & (n.lines.bus0.str[:2] != ct) & (n.lines.bus1.str[:2] == ct)]
        outgoing_line = n.lines.index[(n.lines.carrier == "AC") & (n.lines.bus0.str[:2] == ct) & (n.lines.bus1.str[:2] != ct)]
        
        incoming_link = n.links.index[(n.links.carrier == "DC") & (n.links.bus0.str[:2] != ct) & (n.links.bus1.str[:2] == ct)]
        outgoing_link = n.links.index[(n.links.carrier == "DC") & (n.links.bus0.str[:2] == ct) & (n.links.bus1.str[:2] != ct)]

        incoming_line_p = (n.model["Line-s"].loc[:, incoming_line]*n.snapshot_weightings.generators).sum()
        outgoing_line_p = (n.model["Line-s"].loc[:, outgoing_line]*n.snapshot_weightings.generators).sum()

        incoming_link_p = (n.model["Link-p"].loc[:, incoming_link]*n.snapshot_weightings.generators).sum()
        outgoing_link_p = (n.model["Link-p"].loc[:, outgoing_link]*n.snapshot_weightings.generators).sum()

        lhs = (incoming_link_p - outgoing_link_p) + (incoming_line_p - outgoing_line_p)

        cname = f"Electricity_import_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit, name=f"GlobalConstraint-{cname}"
        )

        if cname not in n.global_constraints.index:
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

    sectors = determine_emission_sectors(n.config['sector'])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in limit_countries:
        limit = co2_total_totals[ct]*limit_countries[ct]
        logger.info(
            f"Limiting emissions in country {ct} to {limit_countries[ct]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

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

        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == "EU renewable oil -> DE oil"]
        outgoing_oil = n.links.index[n.links.index == "DE renewable oil -> EU oil"]

        if not debug:
            lhs.append(
                (-1 * n.model["Link-p"].loc[:, incoming_oil]
                 * 0.2571 * n.snapshot_weightings.generators).sum())
            lhs.append(
                (n.model["Link-p"].loc[:, outgoing_oil]
                 * 0.2571 * n.snapshot_weightings.generators).sum())

        incoming_methanol = n.links.index[n.links.index == "EU methanol -> DE methanol"]
        outgoing_methanol = n.links.index[n.links.index == "DE methanol -> EU methanol"]

        lhs.append(
            (-1 * n.model["Link-p"].loc[:, incoming_methanol]
             / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
             * n.snapshot_weightings.generators).sum())
        
        lhs.append(
            (n.model["Link-p"].loc[:, outgoing_methanol]
             / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
             * n.snapshot_weightings.generators).sum())
        
        # Methane still missing, because its complicated

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname not in n.global_constraints.index:
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


def add_h2_derivate_limit(n, snapshots, investment_year, config):

    for ct in config["limits_volume_max"]["h2_derivate_import"]:
        limit = config["limits_volume_max"]["h2_derivate_import"][ct][investment_year]*1e6

        logger.info(f"limiting H2 derivate imports in {ct} to {limit/1e6} TWh/a")

        incoming = n.links.loc[["EU renewable oil -> DE oil", "EU methanol -> DE methanol"]].index
        outgoing = n.links.loc[["DE renewable oil -> EU oil", "DE methanol -> EU methanol"]].index

        incoming_p = (n.model["Link-p"].loc[:, incoming]*n.snapshot_weightings.generators).sum()
        outgoing_p = (n.model["Link-p"].loc[:, outgoing]*n.snapshot_weightings.generators).sum()

        lhs = incoming_p - outgoing_p
        
        cname = f"H2_derivate_import_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit, name=f"GlobalConstraint-{cname}"
        )
        
        if cname not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname,
                constant=limit,
                sense="<=",
                type="",
                carrier_attribute="",
            )

def force_H2_retrofit(n, force_year, planning_horizon):
    ''''
    Force the retrofit of existing gas plants to H2 from a certain year onwards.
    '''
    logger.info(f"Forcing retrofit of existing gas plants to H2 from {force_year} onwards")
    # Remove extendable gas plants from this planning_horizon
    carriers = ["OCGT", "CCGT", "urban central gas CHP"]
    remove_i = n.links[(n.links.carrier.isin(carriers)) & 
                       (n.links.p_nom_extendable) & 
                       (n.links.bus0.str[:2] == "DE") &
                       (n.links.build_year == planning_horizon)].index
    n.links.drop(remove_i, inplace=True)

    if n.links[n.links.carrier.str.contains("retrofitted H2") & (n.links.bus0.str[:2] == "DE")].empty:
        logger.info("No retrofitted H2 plants in Germany found")
        return

    # Add constraint to force retrofit
    carriers = ["retrofitted H2 OCGT", "retrofitted H2 CCGT", "urban central retrofitted H2 CHP"]

    for carrier in carriers:
        h2_plants = n.links[
            (n.links.carrier == carrier) &
            (n.links.p_nom_extendable) &
            (n.links.build_year < planning_horizon) &
            (n.links.bus0.str[:2] == "DE")
        ].index

        if h2_plants.empty:
            continue

        # Store p_nom value for rhs of constraint
        p_nom = n.model["Link-p_nom"]

        lhs = p_nom.loc[h2_plants]
        rhs = n.links.p_nom_max[h2_plants]
        n.model.add_constraints(lhs <= rhs, name=f"force retrofit of {carrier}")

def additional_functionality(n, snapshots, snakemake):

    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])

    add_min_limits(n, investment_year, snakemake.config)

    add_max_limits(n, investment_year, snakemake.config)

    h2_import_limits(n, snapshots, investment_year, snakemake.config)
    
    electricity_import_limits(n, snapshots, investment_year, snakemake.config)
    
    if investment_year >= 2025:
        h2_production_limits(n, snapshots, investment_year, snakemake.config)
    
    if not snakemake.config["run"]["debug_h2deriv_limit"]:
        add_h2_derivate_limit(n, snapshots, investment_year, snakemake.config)

    #force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    if snakemake.config["sector"]["co2_budget_national"]:
        limit_countries = snakemake.config["co2_budget_national"][investment_year]
        add_co2limit_country(n, limit_countries, snakemake,                  
            debug=snakemake.config["run"]["debug_co2_limit"])

    if snakemake.config["H2_force_retrofit"] <= int(snakemake.wildcards.planning_horizons):
        print(snakemake.wildcards.planning_horizons)
        force_H2_retrofit(n, snakemake.config["H2_force_retrofit"], int(snakemake.wildcards.planning_horizons))