# -*- coding: utf-8 -*-
import os
import sys
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from export_ariadne_variables import hack_transmission_projects

### definitions
year_colors = [
    "dimgrey",
    "darkorange",
    "seagreen",
    "cadetblue",
    "hotpink",
    "darkviolet",
]
markers = [
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "*",
    "+",
    "d",
    "o",
    "|",
    "s",
    "P",
    "p",
    "h",
]
date_format = "%Y-%m-%d %H:%M:%S"

resistive_heater = [
    "urban central resistive heater",
    "rural resistive heater",
    "urban decentral resistive heater",
]
gas_boiler = [
    "urban central gas boiler",
    "rural gas boiler",
    "urban decentral gas boiler",
]
heat_pump = [
    "urban central air heat pump",
    "rural air heat pump",
    "rural ground heat pump",
    "urban decentral air heat pump",
]
water_tanks_charger = [
    "urban central water tanks charger",
    "rural water tanks charger",
    "urban decentral water tanks charger",
]
water_tanks_discharger = [
    "urban central water tanks discharger",
    "rural water tanks discharger",
    "urban decentral water tanks discharger",
]
solar_thermal = [
    "urban decentral solar thermal",
    "urban central solar thermal",
    "rural solar thermal",
]

carrier_renaming = {
    "urban central solid biomass CHP CC": "biomass CHP CC",
    "urban central solid biomass CHP": "biomass CHP",
    "urban central gas CHP": "gas CHP",
    "urban central gas CHP CC": "gas CHP CC",
    "urban central air heat pump": "air heat pump",
    "urban central resistive heater": "resistive heater",
}

carrier_renaming_reverse = {
    "biomass CHP CC": "urban central solid biomass CHP CC",
    "biomass CHP": "urban central solid biomass CHP",
    "gas CHP": "urban central gas CHP",
    "gas CHP CC": "urban central gas CHP CC",
    "air heat pump": "urban central air heat pump",
    "resistive heater": "urban central resistive heater",
}

c1_groups = [
    resistive_heater,
    gas_boiler,
    heat_pump,
    water_tanks_charger,
    water_tanks_discharger,
    solar_thermal,
]
c1_groups_name = [
    "resistive heater",
    "gas boiler",
    "heat pump",
    "water tanks charger",
    "water tanks discharger",
    "solar thermal",
]


def nodal_balance(n, carrier, regions, time=slice(None), aggregate=None, energy=True):
    """
    Calculate energy balance / active power per energy carrier and time steps.

    Arguments:
        carrier: carrier or list of carriers you want to calculate the balance (bus carriers)
        time: time period or list of snapshots as strings e.g. "2013-01-01" or ["2013-01-01 00:00:00", "2013-01-01 03:00:00"]
        aggregate: specify item of ['component', 'snapshot', 'bus', 'carrier'] which will be excluded from Index and aggregated (sum) on it
        energy: if set to true the balance is multiplied by 3 to simulate an aggregation of 24 hours simulation instead of 8 hours simulation (as we have in the network)
    Returns:
        Aggregated active power per carrier, time step, and bus.
    """
    if not isinstance(carrier, list):
        carrier = [carrier]

    if not isinstance(regions, list):
        regions = [regions]

    one_port_data = {}

    for c in n.iterate_components(n.one_port_components):

        df = c.df[
            (c.df.bus.map(n.buses.carrier).isin(carrier))
            & c.df.bus.index.str.startswith(tuple(regions))
        ]

        if df.empty:
            continue

        s = c.pnl.p.loc[time, df.index] * df.sign
        s = s.T.groupby([df.bus.map(n.buses.location), df.carrier]).sum().T
        one_port_data[c.list_name] = s

    branch_data = {}

    for c in n.iterate_components(n.branch_components):
        for col in c.df.columns[c.df.columns.str.startswith("bus")]:

            end = col[3:]
            df = c.df[
                c.df[col].map(n.buses.carrier).isin(carrier)
                & c.df[col].str.startswith(tuple(regions))
            ]

            if df.empty:
                continue

            s = -c.pnl[f"p{end}"].loc[time, df.index]
            s = s.T.groupby([df[col].map(n.buses.location), df.carrier]).sum().T
            branch_data[(c.list_name, end)] = s

    branch_balance = pd.concat(branch_data).groupby(level=[0, 2]).sum()
    one_port_balance = pd.concat(one_port_data)

    def skip_tiny(df, threshold=1e-1):
        return df.where(df.abs() > threshold)

    branch_balance = skip_tiny(branch_balance)
    one_port_balance = skip_tiny(one_port_balance)
    balance = pd.concat([one_port_balance, branch_balance])
    balance = balance.stack(level=[0, 1], future_stack=True)

    balance.index.set_names(["component", "bus"], level=[0, 2], inplace=True)

    if energy:
        balance = balance * n.snapshot_weightings.generators

    if aggregate is not None:
        keep_levels = balance.index.names.difference(aggregate)
        balance = balance.groupby(level=keep_levels).sum()

    return balance


def get_condense_sum(df, groups, groups_name, return_original=False):
    """
    return condensed df, that has been groupeb by condense groups
    Arguments:
        df: df you want to condense (carriers have to be in the columns)
        groups: group lables you want to condense on
        groups_name: name of the new grouped column
        return_original: boolean to specify if the original df should also be returned
    Returns:
        condensed df
    """
    result = df

    for group, name in zip(groups, groups_name):
        # check if carrier are in columns
        bool = [c in df.columns for c in group]
        # updated to carriers within group that are in columns
        group = list(compress(group, bool))

        result[name] = df[group].sum(axis=1)
        result.drop(group, axis=1, inplace=True)

    if return_original:
        return result, df

    return result


def plot_nodal_balance(
    network,
    tech_colors,
    savepath,
    carriers=["AC", "low voltage"],
    start_date="2019-01-01 00:00:00",
    end_date="2019-12-31 00:00:00",
    regions=["DE"],
    model_run="Model run",
    c1_groups=c1_groups,
    c1_groups_name=c1_groups_name,
    loads=["electricity", "industry electricity", "agriculture electricity"],
):

    carriers = carriers
    loads = loads
    start_date = start_date
    end_date = end_date
    regions = regions
    period = network.generators_t.p.index[
        (network.generators_t.p.index >= start_date)
        & (network.generators_t.p.index <= end_date)
    ]
    nb_el = nodal_balance(
        network,
        carrier=carriers,
        regions=regions,
        time=period,
        aggregate=["component", "bus"],
        energy=True,
    )
    i_loads = network.loads[
        network.loads.carrier.isin(loads)
        & network.loads.bus.str.startswith(tuple(regions))
    ].index

    # convert from MW to GW and unstack
    nb_el = nb_el.unstack(level=[1]) / 1000
    loads_el = (
        network.loads_t.p[i_loads].sum(axis=1)
        * network.snapshot_weightings.generators
        / 1000
    )
    nb_el.drop(
        ["electricity distribution grid"], axis=1, inplace=True
    )  # also drop AC if you specifiy no regions (whole system)

    # condense groups
    nb_el = get_condense_sum(nb_el, c1_groups, c1_groups_name)
    # rename unhandy column names
    nb_el.rename(columns=carrier_renaming, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    df = nb_el
    df_loads = loads_el[period]

    # split into df with positive and negative values
    df_neg, df_pos = df.clip(upper=0), df.clip(lower=0)
    # exclude all technologies that contribute less than th
    th = 0.001
    df_pos_share = df_pos.sum() / df_pos.sum().sum()
    df_pos = df_pos[df_pos_share[df_pos_share > th].sort_values(ascending=False).index]
    df_neg_share = df_neg.sum() / df_neg.sum().sum()
    df_neg = df_neg[df_neg_share[df_neg_share > th].sort_values(ascending=False).index]
    # get colors
    c_neg, c_pos = [tech_colors[col] for col in df_neg.columns], [
        tech_colors[col] for col in df_pos.columns
    ]

    # plot positive values
    ax = df_pos.plot.area(ax=ax, stacked=True, color=c_pos, linewidth=0.0)

    # rename negative values that are also present on positive side, so that they are not shown and plot negative values
    f = lambda c: "out_" + c
    cols = [f(c) if (c in df_pos.columns) else c for c in df_neg.columns]
    cols_map = dict(zip(df_neg.columns, cols))
    ax = df_neg.rename(columns=cols_map).plot.area(
        ax=ax, stacked=True, color=c_neg, linewidth=0.0
    )

    # plot lmps
    lmps = network.buses_t.marginal_price[
        network.buses[network.buses.carrier.isin(carriers)].index
    ].mean(axis=1)[period]
    ax2 = lmps.plot(
        style="--", color="black", label="lmp (mean over buses)", secondary_y=True
    )
    ax2.grid(False)
    # set limits of secondary y-axis
    ax2.set_ylim(
        [
            -1.5
            * lmps.max()
            * abs(df_neg.sum(axis=1).min())
            / df_pos.sum(axis=1).max(),
            1.5 * lmps.max(),
        ]
    )

    # plot loads
    df_loads.plot(style=":", color="black", label="electricity loads")

    # explicitly filter out duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if not l.startswith("out_")
    ]
    handles, labels = zip(*filtered_handles_labels)

    # rescale the y-axis
    ax.set_ylim([1.05 * df_neg.sum(axis=1).min(), 1.05 * df_pos.sum(axis=1).max()])
    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(1.22, 1.01),
        title="Legend for left y-axis",
    )
    ax2.legend(title="Legend for right y-axis", loc="upper right")
    ax.set_ylabel("total electriyity balance [GW]")
    ax2.set_ylabel("lmp [€/MWh]")
    ax.set_xlabel("")
    ax.set_title(
        f"Electricity balance (model: {model_run}, period: {start_date} - {end_date})",
        fontsize=16,
        pad=15,
    )
    ax.grid(True)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

    return fig


def plot_storage(
    network,
    tech_colors,
    savepath,
    model_run="Model run",
    start_date="2019-01-01 00:00:00",
    end_date="2019-12-31 00",
    regions=["DE"],
):

    # State of charge [per unit of max] (all stores and storage units)
    # Ratio of total generation of max state of charge

    n = network

    # storage carriers
    st_carriers = [
        "battery",
        "EV battery",
        "PHS",
        "hydro",
        "H2 Store",
    ]  # "battery", "Li ion",
    # generation carriers
    carriers = [
        "battery discharger",
        "V2G",
        "PHS",
        "hydro",
        "H2",
    ]  # "battery discharger", "V2G",
    period = n.generators_t.p.index[
        (n.generators_t.p.index >= start_date) & (n.generators_t.p.index <= end_date)
    ]

    stor_res = pd.DataFrame(
        index=st_carriers,
        columns=[
            "max_charge",
            "gen_charge_ratio",
            "max_stor_cap",
            "gen_charge_cap_ratio",
            "gen_sum",
        ],
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), height_ratios=[1, 1.5])

    for i, c in enumerate(st_carriers):
        if c in n.storage_units.carrier.unique().tolist():
            charge = n.storage_units_t.state_of_charge.loc[
                :, n.storage_units.carrier == c
            ].sum(axis=1)[period]
            gen_sum = (
                n.storage_units_t.p_dispatch.loc[
                    period, n.storage_units.carrier == carriers[i]
                ]
                .sum()
                .sum()
            )
            gen = n.storage_units_t.p_dispatch.loc[
                period, n.storage_units.carrier == carriers[i]
            ].sum(axis=1)
            index = n.storage_units[n.storage_units.carrier == c].index
            max_stor_cap = (n.storage_units.max_hours * n.storage_units.p_nom_opt)[
                index
            ].sum()
            stor_res.loc[c, "max_charge"] = charge.max()
            stor_res.loc[c, "gen_charge_ratio"] = gen_sum / charge.max()
            stor_res.loc[c, "max_stor_cap"] = max_stor_cap
            stor_res.loc[c, "gen_charge_cap_ratio"] = gen_sum / max_stor_cap
            stor_res.loc[c, "gen_sum"] = gen_sum

        elif c in n.stores.carrier.unique().tolist():
            # state of charge (sum over different stores at same location)
            charge = n.stores_t.e.loc[:, n.stores.carrier == c].sum(axis=1)[period]
            gen_sum = (
                -n.links_t.p1.loc[period, n.links.carrier == carriers[i]].sum().sum()
            )
            max_stor_cap = n.stores.e_nom_opt[
                n.stores[n.stores.carrier == c].index
            ].sum()
            stor_res.loc[c, "max_charge"] = charge.max()
            stor_res.loc[c, "gen_charge_ratio"] = gen_sum / charge.max()
            stor_res.loc[c, "max_stor_cap"] = max_stor_cap
            stor_res.loc[c, "gen_charge_cap_ratio"] = gen_sum / max_stor_cap
            stor_res.loc[c, "gen_sum"] = gen_sum

        if c in ["battery", "EV battery", "Li ion"]:
            ax1.plot(
                charge / max_stor_cap,
                label=c,
                color=tech_colors[c],
                marker=markers[i],
                markevery=[0],
                mfc="white",
                mec="black",
            )

        else:
            ax2.plot(
                charge / max_stor_cap,
                label=c,
                color=tech_colors[c],
                marker=markers[i],
                markevery=[0],
                mfc="white",
                mec="black",
            )

    ax1.set_title(f"State of charge of short-term storage technologies ({model_run})")
    ax2.set_title(
        f"State of charge of mid- and long-term storage technologies({model_run})"
    )
    ax1.set_ylabel("State of charge [per unit of max storage capacity]")
    ax2.set_ylabel("State of charge [per unit of max storage capacity]")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    fig.tight_layout(pad=3)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

    return fig


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_ariadne_report",
            simpl="",
            clusters=27,
            opts="",
            ll="vopt",
            sector_opts="None",
            run="KN2045_Bal_v4",
        )

    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons
    post_discretization = snakemake.params.post_discretization
    nhours = int(snakemake.params.hours[:-1])
    nyears = nhours / 8760
    tech_colors = snakemake.params.plotting["tech_colors"]

    for old_name, new_name in carrier_renaming.items():
        if old_name in tech_colors:
            tech_colors[new_name] = tech_colors[old_name]

    # manual carriers
    tech_colors["urban central oil CHP"] = tech_colors["oil"]

    # Load data
    _networks = [pypsa.Network(fn) for fn in snakemake.input.networks]
    modelyears = [fn[-7:-3] for fn in snakemake.input.networks]
    # Hack the transmission projects
    networks = [
        hack_transmission_projects(n.copy(), int(my))
        for n, my in zip(_networks, modelyears)
    ]

    # electricity supply and demand

    plot_nodal_balance(
        network=networks[4],
        tech_colors=tech_colors,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        savepath=snakemake.output.elec_supply_demand_total,
        model_run=snakemake.wildcards.run,
    )

    plot_nodal_balance(
        network=networks[4],
        tech_colors=tech_colors,
        start_date="2019-01-01 00:00:00",
        end_date="2019-01-31 00:00:00",
        savepath=snakemake.output.elec_supply_demand_A,
        model_run=snakemake.wildcards.run,
    )

    plot_nodal_balance(
        network=networks[4],
        tech_colors=tech_colors,
        start_date="2019-05-01 00:00:00",
        end_date="2019-05-31 00:00:00",
        savepath=snakemake.output.elec_supply_demand_B,
        model_run=snakemake.wildcards.run,
    )

    # storage

    plot_storage(
        network=networks[4],
        tech_colors=tech_colors,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        savepath=snakemake.output.storage,
        model_run=snakemake.wildcards.run,
    )
