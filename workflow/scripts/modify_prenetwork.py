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
    n.add("Bus", "DE", carrier="none")
    n.add("Bus", "DE oil", carrier="oil")
    n.add("Bus", "DE renewable oil", carrier="renewable oil")
    n.add(
        "Bus", 
        "EU renewable oil",
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

def unravel_import_carrier(n, industrial_demand, industrial_production):

    logger.info("Unraveling steel and hbi import")
    # needed for the loads
    endogenous_sectors = []
    options = snakemake.config["sector"]
    if options["endogenous_steel"]:
        endogenous_sectors += ["DRI + Electric arc"]
    if options["endogenous_hvc"]:
        endogenous_sectors += ["HVC"]
    sectors_b = ~industrial_demand.index.get_level_values("sector").isin(
        endogenous_sectors
    )
    HVC_demand_factor = options.get("HVC_demand_factor", 1)

    if any("shipping-steel" in keys for keys in snakemake.params.sector["import"]["options"]):
        ###
        # add hbi/steel bus
        n.add("Bus", "DE hbi", carrier="hbi")
        n.add("Bus", "DE steel", carrier="steel")

        # hbi links
        hbi_links = n.links[(n.links.carrier=="hbi") & (n.links.index.str.contains("DE"))].index
        n.links.loc[hbi_links, "bus1"] = "DE steel"
        n.links.loc[hbi_links, "bus2"] = "DE hbi"

        # hbi generator
        hbi_gen = n.generators.loc["EU import shipping-hbi"].copy()
        hbi_gen.name = "DE import shipping-hbi"
        hbi_gen.bus = "DE hbi"
        n.add("Generator", name=hbi_gen.name, **hbi_gen)

        # steel generator
        steel_gen = n.generators.loc["EU import shipping-steel"].copy()
        steel_gen.name = "DE import shipping-steel"
        steel_gen.bus = "DE steel"
        n.add("Generator", name=steel_gen.name, **steel_gen)

        # steel links
        steel_links = n.links[(n.links.carrier=="DRI + Electric arc") & (n.links.index.str.contains("DE"))].index
        n.links.loc[steel_links, "bus1"] = "DE steel"
        
        # transport links
        n.madd(
            "Link",
            ["EU steel -> DE steel", "DE steel -> EU steel"],
            bus0=["EU steel", "DE steel"],
            bus1=["DE steel", "EU steel"],
            carrier="steel",
            p_nom=1e6,
            p_min_pu=0,
        )
        # steel load
        DE_steel = industrial_production["DRI + Electric arc"].filter(like="DE1 ").sum() / nhours
        n.add("Load", "DE steel", bus="DE steel", p_set=DE_steel)
        n.loads.loc["EU steel", "p_set"] -= DE_steel
        n.loads.rename(index={"EU steel": "EUminusDE steel"}, inplace=True)

    ###
    # add ammonia
    logger.info("Unraveling ammonia import")
    n.add("Bus", "DE NH3", carrier="NH3")

    # add ammonia store
    nh3_index = "EU NH3 ammonia store-" + snakemake.wildcards.planning_horizons
    nh3_store = n.stores.loc[nh3_index].copy()
    if ~nh3_store.empty:
        nh3_store.name = "DE NH3 ammonia store-" + snakemake.wildcards.planning_horizons
        nh3_store.bus = "DE NH3"
        n.add("Store", name=nh3_store.name, **nh3_store)

    # ammonia links
    HB_links = n.links[(n.links.carrier=="Haber-Bosch") & (n.links.index.str.contains("DE"))].index
    n.links.loc[HB_links, "bus1"] = "DE NH3"
    crack_links = n.links[(n.links.carrier=="ammonia cracker") & (n.links.index.str.contains("DE"))].index
    n.links.loc[crack_links, "bus0"] = "DE NH3"

    if any("shipping-lnh3" in keys for keys in snakemake.params.sector["import"]["options"]):
        # ammonia generator
        nh3_gen = n.generators.loc["EU import shipping-lnh3"].copy()
        nh3_gen.name = "DE import shipping-lnh3"
        nh3_gen.bus = "DE NH3"
        n.add("Generator", name=nh3_gen.name, **nh3_gen)

    # transport links
    n.madd(
        "Link",
        ["EU NH3 -> DE NH3", "DE NH3 -> EU NH3"],
        bus0=["EU NH3", "DE NH3"],
        bus1=["DE NH3", "EU NH3"],
        carrier="NH3",
        p_nom=1e6,
        p_min_pu=0,
    )
    # add load
    EU_ammonia = industrial_demand.loc[sectors_b, "ammonia"]
    DE_ammonia = EU_ammonia.loc[EU_ammonia.index.get_level_values("node").str.startswith("DE1")].sum() / nhours
    n.add("Load", "DE NH3 load", bus="DE NH3", p_set=DE_ammonia)
    n.loads.loc["EU NH3", "p_set"] -= DE_ammonia
    n.loads.rename(index={"EU NH3": "EUminusDE NH3"}, inplace=True)

    ###
    # add meoh
    logger.info("Unraveling methanol import")

    n.add("Bus", "DE methanol", carrier="methanol")
    n.add("Bus", "DE HVC", carrier="HVC")

    # add meoh store
    meoh_store = n.stores.loc["EU methanol Store"].copy()
    meoh_store.name = "DE methanol Store"
    meoh_store.bus = "DE methanol"
    n.add("Store", name=meoh_store.name, **meoh_store)

    if any("shipping-meoh" in keys for keys in snakemake.params.sector["import"]["options"]):
        n.add("Bus", "DE shipping-meoh", carrier="methanol")
        # meoh generator link
        meoh_gen = n.links.loc["EU import shipping-meoh"].copy()
        meoh_gen.name = "DE import shipping-meoh"
        meoh_gen.bus0 = "DE shipping-meoh"
        meoh_gen.bus1 = "DE methanol"
        n.add("Link", name=meoh_gen.name, **meoh_gen)

    # add meoh links
    industry = n.links[(n.links.carrier=="industry methanol") & (n.links.index.str.contains("DE"))].index
    n.links.loc[industry, "bus0"] = "DE methanol"

    allam = n.links[(n.links.carrier=="allam methanol") & (n.links.index.str.contains("DE"))].index
    n.links.loc[allam, "bus0"] = "DE methanol"

    HVC = n.links[(n.links.carrier=="methanol-to-olefins/aromatics") & (n.links.index.str.contains("DE"))].index
    n.links.loc[HVC, "bus0"] = "DE methanol"
    n.links.loc[HVC, "bus1"] = "DE HVC"

    CCGT = n.links[(n.links.carrier=="CCGT methanol") & (n.links.index.str.contains("DE"))].index
    n.links.loc[CCGT, "bus0"] = "DE methanol"

    CCGT_CC = n.links[(n.links.carrier=="CCGT methanol CC") & (n.links.index.str.contains("DE"))].index
    n.links.loc[CCGT_CC, "bus0"] = "DE methanol"

    OCGT = n.links[(n.links.carrier=="OCGT methanol") & (n.links.index.str.contains("DE"))].index
    n.links.loc[OCGT, "bus0"] = "DE methanol"

    aviation = n.links[(n.links.carrier=="methanol-to-kerosene") & (n.links.index.str.contains("DE"))].index
    n.links.loc[aviation, "bus0"] = "DE methanol"

    methanolisation = n.links[(n.links.carrier=="methanolisation") & (n.links.index.str.contains("DE")) & n.links.index.str[:-4] == snakemake.wildcards.planning_horizons].index
    n.links.loc[methanolisation, "bus1"] = "DE methanol"

    # transport links
    n.madd(
        "Link",
        ["EU methanol -> DE methanol", "DE methanol -> EU methanol"],
        bus0=["EU methanol", "DE methanol"],
        bus1=["DE methanol", "EU methanol"],
        carrier="methanol",
        p_nom=1e6,
        p_min_pu=0,
    )
    
    # add load
    # HVC demand
    DE_HVC = HVC_demand_factor * industrial_production["HVC"].filter(like="DE1 ").sum() / nhours
    n.add("Load", "DE HVC", bus="DE HVC", p_set=DE_HVC)
    n.loads.loc["EU HVC", "p_set"] -= DE_HVC
    n.loads.rename(index={"EU HVC": "EUminusDE HVC"}, inplace=True)

    # global shipping information
    options = snakemake.params.sector
    domestic_navigation = (
    pd.read_csv(snakemake.input.pop_weighted_energy_totals, index_col=0) * nyears
    ).loc[:, "total domestic navigation"].squeeze()
    international_navigation = pd.read_csv(snakemake.input.shipping_demand, index_col=0).squeeze() * nyears
    shipping_demand = domestic_navigation + international_navigation
    DE_shipping = shipping_demand.filter(like="DE").sum() * 1e6 / nhours
    
    # shipping meoh
    if "EU methanol shipping methanol" in n.loads.index:
        efficiency = (
            options["shipping_oil_efficiency"] / options["shipping_methanol_efficiency"]
        )
        shipping_methanol_share = options["shipping_methanol_share"][int(snakemake.wildcards.planning_horizons)]

        p_set_methanol = shipping_methanol_share * DE_shipping * efficiency

        n.add("Load", "DE shipping methanol", bus="DE methanol", carrier="methanol" , p_set=p_set_methanol)
        n.loads.loc["EU methanol shipping methanol", "p_set"] -= p_set_methanol

        # CO2 intensity methanol based on stoichiometric calculation with 22.7 GJ/t methanol (32 g/mol), CO2 (44 g/mol), 277.78 MWh/TJ = 0.218 t/MWh
        co2 = p_set_methanol * costs.at["methanolisation", "carbondioxide-input"]

        n.add(
            "Load",
            "DE shipping methanol emissions",
            bus="co2 atmosphere",
            carrier="shipping methanol emissions",
            p_set=-co2,
        )
        n.loads.loc["shipping methanol emissions", "p_set"] += co2

    # industry meoh demand
    EU_meoh = industrial_demand["methanol"]
    DE_meoh = EU_meoh.loc[EU_meoh.index.get_level_values("node").str.startswith("DE1")].sum() / nhours
    n.add("Load", "DE industry methanol", bus="DE methanol", p_set=DE_meoh)
    n.loads.loc["EU industry methanol", "p_set"] -= DE_meoh
    n.loads.rename(index={"EU industry methanol": "EUminusDE industry methanol"}, inplace=True)

    ###
    # add ft
    if any("shipping-ftfuel" in keys for keys in snakemake.params.sector["import"]["options"]):
        logger.info("Unraveling Fischer-Tropsch import")

        n.add("Bus", "DE shipping-ftfuel", carrier="ftfuel")
        n.add("Bus", "DE ftfuel", carrier="ftfuel")

        # add ft store
        ft_store = n.stores.loc["EU import shipping-ftfuel store"].copy()
        ft_store.name = "DE ftfuel Store"
        ft_store.bus = "DE ftfuel"
        n.add("Store", name=ft_store.name, **ft_store)

        # ft generator link
        ft_gen = n.links.loc["EU import shipping-ftfuel"].copy()
        ft_gen.name = "DE import shipping-ftfuel"
        ft_gen.bus0 = "DE shipping-ftfuel"
        ft_gen.bus1 = "DE renewable oil"
        n.add("Link", name=ft_gen.name, **ft_gen)

    # oil shipping
    if "EU oil shipping oil" in n.loads.index:
        shipping_oil_share = options["shipping_oil_share"][int(snakemake.wildcards.planning_horizons)]

        p_set_oil = shipping_oil_share * DE_shipping

        n.add("Load", "DE shipping oil", bus="DE oil", carrier="oil", p_set=p_set_oil,)
        n.loads.loc["EU oil shipping oil", "p_set"] -= p_set_oil

        co2 = p_set_oil * costs.at["oil", "CO2 intensity"]

        n.add("Load", "DE shipping oil emissions", bus="co2 atmosphere", carrier="shipping oil emissions", p_set=-co2)
        n.loads.loc["shipping oil emissions", "p_set"] += co2
    
    # everything else is already taken care of in unravel_oilbus
    # delete all European import links/generators
    drop_gen = n.generators[(n.generators.index.str.contains("shipping")) & 
                            (n.generators.bus.str[:2] != "DE")].index
    carriers = ["import shipping-lch4", "import shipping-ftfuel", "import shipping-meoh"]
    drop_lin = n.links[(n.links.carrier.isin(carriers)) & 
                       (n.links.index.str[:2] != "DE")].index

    n.generators.drop(drop_gen, inplace=True)
    n.links.drop(drop_lin, inplace=True)


def transmission_costs_from_modified_cost_data(n, costs, transmission, length_factor=1.0):
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

    if transmission == "overhead":
        links_costs = "HVDC overhead"
    elif transmission == "underground":
        links_costs = "HVDC submarine"

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at[links_costs, "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"]
            * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs

def must_run_biomass(n, p_min_pu, regions):
    """
    Set p_min_pu for biomass generators to the specified value.
    """
    logger.info(f"Must-run condition enabled: Setting p_min_pu = {p_min_pu} for biomass generators.")
    links_i = n.links[(n.links.carrier == 'solid biomass') & (n.links.bus0.str.startswith(tuple(regions)))].index
    n.links.loc[links_i, "p_min_pu"] = p_min_pu


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
            clusters=37,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2030",
            run="cost_plus_10",
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

    # fix_new_boiler_profiles(n)

    remove_old_boiler_profiles(n)

    coal_generation_ban(n)
    
    nuclear_generation_ban(n)

    first_technology_occurrence(n)

    if not snakemake.config["run"]["debug_unravel_oilbus"]:
        unravel_oilbus(n)
        if snakemake.params.sector["imp"]:
            industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=[0, 1]) * 1e6 * nyears
            industrial_production = pd.read_csv(snakemake.input.industrial_production, index_col=0) * 1e3 * nyears  # kt/a -> t/a
            unravel_import_carrier(n, industrial_demand, industrial_production)

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
    transmission_costs_from_modified_cost_data(n, costs_loaded, snakemake.params.transmission_costs, snakemake.params.length_factor)

    if snakemake.params.biomass_must_run["enable"]:
        must_run_biomass(n, snakemake.params.biomass_must_run["p_min_pu"], snakemake.params.biomass_must_run["regions"])

    n.export_to_netcdf(snakemake.output.network)
