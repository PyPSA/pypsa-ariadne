# -*- coding: utf-8 -*-
import os
import sys
import logging
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import geopandas as gpd
import cartopy.crs as ccrs

from matplotlib.lines import Line2D
from pypsa.plot import add_legend_lines
from itertools import compress
from multiprocessing import Pool

path = "../submodules/pypsa-eur/scripts"
sys.path.insert(1, os.path.abspath(path))
# from export_ariadne_variables import hack_transmission_projects
from _helpers import configure_logging, set_scenario_config
from plot_summary import preferred_order, rename_techs
from plot_power_network import load_projection
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

logger = logging.getLogger(__name__)

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
air_heat_pump = [
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
solar = [
    "solar rooftop",
    "solar-hsat",
    "solar",
]
electricity_load = [
    "electricity",
    "industry electricity",
    "agriculture electricity",
]
electricity_imports = [
    "AC",
    "DC",
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
    air_heat_pump,
    water_tanks_charger,
    water_tanks_discharger,
    solar_thermal,
    solar,
    electricity_load,
    electricity_imports,
]
c1_groups_name = [
    "resistive heater",
    "gas boiler",
    "air heat pump",
    "water tanks charger",
    "water tanks discharger",
    "solar thermal",
    "solar",
    "electricity load",
    "electricity import"
]

### functions

def hack_transmission_projects(n, model_year):
    print("Hacking transmission projects for year", model_year)
    tprojs = n.links.loc[
        (n.links.index.str.startswith("DC") | n.links.index.str.startswith("TYNDP"))
        & ~n.links.reversed
    ].index

    future_projects = tprojs[n.links.loc[tprojs, "build_year"] > model_year]
    current_projects = tprojs[
        (n.links.loc[tprojs, "build_year"] > (model_year - 5))
        & (n.links.loc[tprojs, "build_year"] <= model_year)
    ]
    past_projects = tprojs[n.links.loc[tprojs, "build_year"] <= (model_year - 5)]

    # Future projects should not have any capacity
    # assert isclose(n.links.loc[future_projects, "p_nom_opt"], 0).all()
    # Setting p_nom to 0 such that n.statistics does not compute negative expanded capex or capacity additions
    # Setting p_nom to 0 for the grid_expansion calculation
    # This is ONLY POSSIBLE IN POST-PROCESSING
    # We pretend that the model expanded the grid endogenously
    n.links.loc[future_projects, "p_nom"] = 0
    n.links.loc[future_projects, "p_nom_min"] = 0

    # Current projects should have their p_nom_opt equal to p_nom
    # until the year 2030 (Startnetz that we force in)
    if model_year <= 2030:
        # assert isclose(
        #     n.links.loc[current_projects, "p_nom_opt"],
        #     n.links.loc[current_projects, "p_nom"],
        # ).all()

        n.links.loc[current_projects, "p_nom"] = 0
        n.links.loc[current_projects, "p_nom_min"] = 0

    else:
        n.links.loc[current_projects, "p_nom"] = n.links.loc[
            current_projects, "p_nom_min"
        ]

    # Past projects should have their p_nom_opt bigger or equal to p_nom
    # if model_year <= 2035:
    #     # assert (
    #     #     n.links.loc[past_projects, "p_nom_opt"]
    #     #     >= n.links.loc[past_projects, "p_nom"]
    #     # ).all()

    return n

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
    plot_lmps=True,
    plot_loads=True,
    resample=None,
    nice_names=False,
    threshold=0,
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
    nb = nodal_balance(
        network,
        carrier=carriers,
        regions=regions,
        time=period,
        aggregate=["component", "bus"],
        energy=True,
    )

    # convert from MW to GW and unstack
    nb = nb.unstack(level=[1]) / 1e3

    # condense groups
    nb = get_condense_sum(nb, c1_groups, c1_groups_name)
    # rename unhandy column names
    nb.rename(columns=carrier_renaming, inplace=True)

    ## summaris low contributing carriers as other
    techs_below_threshold = df.columns[df.abs().max() < threshold].tolist()
    if techs_below_threshold:
        other = {tech: "other" for tech in techs_below_threshold}
        rename.update(other)
        colors["other"] = "grey"

    if rename:
        df = df.groupby(df.columns.map(lambda a: rename.get(a, a)), axis=1).sum()

    if resample is not None:
        nb = nb.resample(resample).mean()

    df = nb
    # split into df with positive and negative values
    df_neg, df_pos = df.clip(upper=0), df.clip(lower=0)
    # summarise all technologie which contribute less than th as other

    # exclude all technologies that contribute less than th (GWh)
    th = 0.001
    df_pos_share = df_pos.sum() / df_pos.sum().sum()
    df_pos = df_pos[df_pos_share[df_pos_share > th].sort_values(ascending=False).index]
    df_neg_share = df_neg.sum() / df_neg.sum().sum()
    df_neg = df_neg[df_neg_share[df_neg_share > th].sort_values(ascending=False).index]
    # get colors
    c_neg, c_pos = [tech_colors[col] for col in df_neg.columns], [
        tech_colors[col] for col in df_pos.columns
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

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
    if plot_lmps:   
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
        ax2.legend(title="Legend for right y-axis", loc="upper right")
        ax2.set_ylabel("lmp [€/MWh]")

    # plot loads
    if plot_loads:
        i_loads = network.loads[
        network.loads.carrier.isin(loads)
        & network.loads.bus.str.startswith(tuple(regions))
        ].index
        loads = (
            network.loads_t.p[i_loads].sum(axis=1)
            * network.snapshot_weightings.generators
            / 1e3
        )
        df_loads = loads[period]
        df_loads.plot(style=":", color="black", label="loads")

    # explicitly filter out duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if not l.startswith("out_")
    ]
    handles, labels = zip(*filtered_handles_labels)

    if nice_names:
        nice_names_dict = network.carriers.nice_name.to_dict()
        labels = [nice_names_dict.get(l, l) for l in labels]

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

    ax.set_ylabel("total electriyity balance [GW]")
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


### Fabians plotting routine

THRESHOLD = 5  # GW

CARRIER_GROUPS = {
    "electricity": ["AC", "low voltage"],
    "heat": [
        "urban central heat",
        "urban decentral heat",
        "rural heat",
        "residential urban decentral heat",
        "residential rural heat",
        "services urban decentral heat",
        "services rural heat",
    ],
    # "hydrogen": "H2",
    # "oil": "oil",
    # "methanol": "methanol",
    # "ammonia": "NH3",
    # "biomass": ["solid biomass", "biogas"],
    # "CO2 atmosphere": "co2",
    # "CO2 stored": "co2 stored",
    # "methane": "gas",
}


def plot_stacked_area_steplike(ax, df, colors={}):
    if isinstance(colors, pd.Series):
        colors = colors.to_dict()

    df_cum = df.cumsum(axis=1)

    previous_series = np.zeros_like(df_cum.iloc[:, 0].values)

    for col in df_cum.columns:
        ax.fill_between(
            df_cum.index,
            previous_series,
            df_cum[col],
            step="pre",
            linewidth=0,
            color=colors.get(col, "grey"),
            label=col,
        )
        previous_series = df_cum[col].values


def plot_energy_balance_timeseries(
    df,
    time=None,
    ylim=None,
    resample=None,
    rename={},
    preferred_order=[],
    ylabel="",
    colors={},
    threshold=0,
    dir="",
):
    if time is not None:
        df = df.loc[time]

    timespan = df.index[-1] - df.index[0]
    long_time_frame = timespan > pd.Timedelta(weeks=5)

    techs_below_threshold = df.columns[df.abs().max() < threshold].tolist()

    if techs_below_threshold:
        other = {tech: "other" for tech in techs_below_threshold}
        rename.update(other)
        colors["other"] = "grey"

    if rename:
        df = df.groupby(df.columns.map(lambda a: rename.get(a, a)), axis=1).sum()

    if resample is not None:
        # upsampling to hourly resolution required to handle overlapping block
        df = df.resample("1H").ffill().resample(resample).mean()

    order = (df / df.max()).var().sort_values().index
    if preferred_order:
        order = preferred_order.intersection(order).append(
            order.difference(preferred_order)
        )
    df = df.loc[:, order]

    # fillna since plot_stacked_area_steplike cannot deal with NaNs
    pos = df.where(df > 0).fillna(0.0)
    neg = df.where(df < 0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    plot_stacked_area_steplike(ax, pos, colors)
    plot_stacked_area_steplike(ax, neg, colors)

    plt.xlim((df.index[0], df.index[-1]))

    if not long_time_frame:
        # Set major ticks every Monday
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        # Set minor ticks every day
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))
    else:
        # Set major ticks every first day of the month
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        # Set minor ticks every 15th of the month
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))

    ax.tick_params(axis="x", which="minor", labelcolor="grey")
    ax.grid(axis="y")

    # half the labels because pos and neg create duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    half = int(len(handles) / 2)
    fig.legend(handles=handles[:half], labels=labels[:half], loc="outside right upper")

    ax.axhline(0, color="grey", linewidth=0.5)

    if ylim is None:
        # ensure y-axis extent is symmetric around origin in steps of 100 units
        ylim = np.ceil(max(-neg.sum(axis=1).min(), pos.sum(axis=1).max()) / 100) * 100
    plt.ylim([-ylim, ylim])

    is_kt = any(s in ylabel.lower() for s in ["co2", "steel", "hvc"])
    unit = "kt/h" if is_kt else "GW"
    plt.ylabel(f"{ylabel} balance [{unit}]")

    if not long_time_frame:
        # plot frequency of snapshots on top x-axis as ticks
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(df.index)
        ax2.grid(False)
        ax2.tick_params(axis="x", length=2)
        ax2.xaxis.set_tick_params(labelbottom=False)
        ax2.set_xticklabels([])

    if resample is None:
        resample = f"native-{time}"
    fn = f"ts-balance-{ylabel.replace(' ', '_')}-{resample}"
    # plt.savefig(dir + "/" + fn + ".pdf")
    plt.savefig(dir + "/" + fn + ".png")
    plt.close()


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


def plot_price_duration_curve(
    networks,
    year_colors,
    savepath,
    years,
    carriers=["AC", "low voltage"],
    aggregate=True,
    model_run="Model run",
    regions=["DE"],
    y_lim_values=[-50, 300],
):

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    for i, n in enumerate(networks.values()):
        buses = n.buses[
            (n.buses.carrier.isin(carriers))
            & n.buses.index.str.startswith(tuple(regions))
        ].index
        lmps = pd.DataFrame(n.buses_t.marginal_price[buses])
        if aggregate:
            lmps = pd.DataFrame(lmps.mean(axis=1))
        else:
            lmps = pd.DataFrame(lmps.values.flatten())

        lmps.columns = ["lmp"]
        lmps.sort_values(by="lmp", ascending=False, inplace=True)
        lmps["percentage"] = np.arange(len(lmps)) / len(lmps) * 100
        ax.plot(lmps["percentage"], lmps["lmp"], label=years[i], color=year_colors[i])

        ax.set_ylim(y_lim_values)
        # # add corridor which contains 75 % of the generation around the median
        # ax.hlines(df["lmp"].loc[df["lmp"][df["gen_cumsum_norm"] > 0.125].index[0]], 0, 1, color=year_colors[i], ls="--", lw=1)
        # ax.hlines(df["lmp"].loc[df["lmp"][df["gen_cumsum_norm"] > 0.875].index[0]], 0, 1,  color=year_colors[i], ls="--", lw =1)

        ax.set_ylabel("Electricity Price [$€/MWh_{el}$")
        ax.set_xlabel("Fraction of time [%]")
        ax.set_title(f"Electricity price duration curves {model_run}", fontsize=16)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

    return fig


def plot_price_duration_hist(
    networks,
    year_colors,
    savepath,
    years,
    carriers=["AC", "low voltage"],
    aggregate=True,
    model_run="Model run",
    regions=["DE"],
    x_lim_values=[-50, 300],
):

    fig, axes = plt.subplots(ncols=1, nrows=len(years), figsize=(8, 3 * len(years)))
    axes = axes.flatten()

    for i, n in enumerate(networks.values()):
        buses = n.buses[
            (n.buses.carrier.isin(carriers))
            & n.buses.index.str.startswith(tuple(regions))
        ].index
        lmps = pd.DataFrame(n.buses_t.marginal_price[buses])
        if aggregate:
            lmps = pd.DataFrame(lmps.mean(axis=1))
        else:
            lmps = pd.DataFrame(lmps.values.flatten())

        lmps.columns = ["lmp"]
        axes[i].hist(lmps, 
                     bins=100, 
                     color=year_colors[i], 
                     alpha=0.5, 
                     label=years[i],
                     range=x_lim_values,)
        axes[i].legend()

    axes[i].set_xlabel("Electricity Price [$€/MWh_{el}$")
    plt.suptitle(f"Electricity prices ({model_run})", fontsize=16, y=0.99)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

    return fig

### plot power network.py

def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif "external" in tech:
        return "import hvdc-to-elec"
    elif tech in ["H2 Electrolysis", "methanation", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #     return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def plot_map(
    n,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=2e10,
    transmission=False,
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]

    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    costs = costs.T.groupby(costs.columns).sum().T

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.06),
        labelspacing=0.8,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.06),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.52, 1.04),
        frameon=False,
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    # fig.savefig(snakemake.output.map, bbox_inches="tight")


### hydrogen network

def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df["index_orig"] = df.index
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


def plot_h2_map(n, 
                regions, 
                savepath,
                only_de=False):

    assign_location(n)

    h2_storage = n.stores[n.stores.carrier.isin(["H2", "H2 Store"])]
    regions["H2"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e6)
    )  # TWh
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 1e5
    linewidth_factor = 4e3
    # MW below which not drawn
    line_lower_threshold = 0

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]
    h2_kern = n.links[n.links.carrier == "H2 pipeline (Kernnetz)"]

    # sum capacitiy for pipelines from different investment periods
    h2_new = group_pipes(h2_new)

    if not h2_retro.empty:
        h2_retro = (
            group_pipes(h2_retro, drop_direction=True)
            .reindex(h2_new.index)
            .fillna(0)
        )

    if not h2_kern.empty:
        h2_kern = (
            group_pipes(h2_kern, drop_direction=True)
            .reindex(h2_new.index)
            .fillna(0)
        )

    h2_total = n.links.p_nom_opt.groupby(level=0).sum()
    link_widths_total = h2_total / linewidth_factor

    # drop all reversed pipe
    n.links.drop(n.links.index[n.links.index.str.contains("reversed")], inplace=True)
    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).agg({
        **{col: 'first' for col in n.links.columns if col != 'p_nom_opt'},  # Take first value for all columns except 'p_nom_opt'
        'p_nom_opt': 'sum'  # Sum values for 'p_nom_opt'
    })
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    carriers_pipe = ["H2 pipeline", "H2 pipeline retrofitted", "H2 pipeline (Kernnetz)"]
    total = n.links.p_nom_opt.where(
        n.links.carrier.isin(carriers_pipe), other=0.0
    )
    
    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )

    kern = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline (Kernnetz)", other=0.0
    )

    link_widths_total = total / linewidth_factor
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_kern = kern / linewidth_factor
    link_widths_kern[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"
    color_kern = "#6b3161"

    bus_colors = {"H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#805394"}

    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_kern,
        link_widths=link_widths_kern,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=6,
        vmin=0,
        legend_kwds={
            "label": "Hydrogen Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.23, 1),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = [bus_colors[c] for c in carriers] + [
        color_h2_pipe,
        color_retrofit,
        color_kern,
    ]
    labels = carriers + [
        "H2 pipeline (total)",
        "H2 pipeline (repurposed)",
        "H2 pipeline (Kernnetz)",
    ]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1.13),
        ncol=2,
        frameon=False,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")

    fig.savefig(savepath, bbox_inches="tight")


def plot_h2_map_de(n, 
                regions, 
                tech_colors,
                savepath,
                specify_buses=None):

    assign_location(n)

    h2_storage = n.stores[n.stores.carrier.isin(["H2", "H2 Store"])]
    regions["H2"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e6)
    )  # TWh
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    
    linewidth_factor = 4e3
    # MW below which not drawn
    line_lower_threshold = 0

    # buses and size
    if specify_buses is None:
        bus_size_factor = 1e5
        carriers = ["H2 Electrolysis", "H2 Fuel Cell"]
        elec = n.links[(n.links.carrier.isin(carriers)) & (n.links.index.str.contains("DE"))].index
        bus_sizes = (
            n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
            / bus_size_factor
        )
    if specify_buses == "production":
        bus_size_factor = 2e8
        h2_producers = n.links.index[n.links.index.str.startswith("DE") & (n.links.bus1.map(n.buses.carrier) == "H2")]
        carriers = h2_producers.map(n.links.carrier).unique().tolist()
        production = -n.links_t.p1[h2_producers].multiply(n.snapshot_weightings.generators,axis=0)
        bus_sizes = production.sum().groupby(
            [production.sum().index.map(n.links.bus1) , production.sum().index.map(n.links.carrier)]
            ).sum() / bus_size_factor

    if specify_buses == "consumption":
        bus_size_factor = 2e8
        # links
        h2_consumers_links = n.links.index[n.links.index.str.startswith("DE") & (n.links.bus0.map(n.buses.carrier) == "H2")]
        consumption_links = n.links_t.p0[h2_consumers_links].multiply(n.snapshot_weightings.generators,axis=0)
        bus_sizes_links = consumption_links.sum().groupby(
            [consumption_links.sum().index.map(n.links.bus0) , consumption_links.sum().index.map(n.links.carrier)]
            ).sum() / bus_size_factor
        #loads
        h2_consumers_loads = n.loads.index[n.loads.bus.str.startswith("DE") & (n.loads.bus.map(n.buses.carrier) == "H2")]
        consumption_loads = n.loads_t.p[h2_consumers_loads].multiply(n.snapshot_weightings.generators,axis=0)
        bus_sizes_loads = consumption_loads.sum().groupby(
            [consumption_loads.sum().index.map(n.loads.bus) , consumption_loads.sum().index.map(n.loads.carrier)]
            ).sum() / bus_size_factor
        
        bus_sizes = pd.concat([bus_sizes_links, bus_sizes_loads])

        def rename_carriers(carrier):
            if "H2" in carrier and "OCGT" in carrier:
                return "H2 OCGT"
            elif "H2" in carrier and "CHP" in carrier:
                return "H2 CHP"
            else:
                return carrier
        bus_sizes = bus_sizes.rename(index=lambda x: rename_carriers(x), level=1)
        bus_sizes = bus_sizes.groupby(level=[0, 1]).sum()
        tech_colors["H2 CHP"] = "darkorange"
        # only select 4 most contributing carriers and summarise rest as other
        others = (bus_sizes.groupby(level=[1]).sum() / bus_sizes.sum()).sort_values(ascending=False)[5:].index.tolist()
        replacement_dict = {value: "other" for value in others}
        bus_sizes = bus_sizes.rename(index=replacement_dict, level=1)
        bus_sizes = bus_sizes.groupby(level=[0, 1]).sum()
        carriers =  bus_sizes.index.get_level_values(1).unique().tolist()
        
    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    # drop all links which are not H2 pipelines or not in germany
    n.links.drop(
        n.links.index[~((n.links['carrier'].str.contains("H2 pipeline")) & (n.links.index.str.contains("DE")))], 
        inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]
    h2_kern = n.links[n.links.carrier == "H2 pipeline (Kernnetz)"]

    # sum capacitiy for pipelines from different investment periods
    h2_new = group_pipes(h2_new)

    if not h2_retro.empty:
        h2_retro = (
            group_pipes(h2_retro, drop_direction=True)
            .reindex(h2_new.index)
            .fillna(0)
        )

    if not h2_kern.empty:
        h2_kern = (
            group_pipes(h2_kern, drop_direction=True)
            .reindex(h2_new.index)
            .fillna(0)
        )

    h2_total = n.links.p_nom_opt.groupby(level=0).sum()
    link_widths_total = h2_total / linewidth_factor

    # drop all reversed pipe
    n.links.drop(n.links.index[n.links.index.str.contains("reversed")], inplace=True)
    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).agg({
        **{col: 'first' for col in n.links.columns if col != 'p_nom_opt'},  # Take first value for all columns except 'p_nom_opt'
        'p_nom_opt': 'sum'  # Sum values for 'p_nom_opt'
    })
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    carriers_pipe = ["H2 pipeline", "H2 pipeline retrofitted", "H2 pipeline (Kernnetz)"]
    total = n.links.p_nom_opt.where(
        n.links.carrier.isin(carriers_pipe), other=0.0
    )
    
    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )

    kern = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline (Kernnetz)", other=0.0
    )

    link_widths_total = total / linewidth_factor
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_kern = kern / linewidth_factor
    link_widths_kern[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"
    color_kern = "#6b3161"

    
    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=tech_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_kern,
        link_widths=link_widths_kern,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=6,
        vmin=0,
        legend_kwds={
            "label": "Hydrogen Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    # Set geographic extent for Germany
    ax.set_extent([5.5, 15.5, 48.0, 56], crs=ccrs.PlateCarree())  # Germany bounds

    if specify_buses is None:
        sizes = [5, 1]
        labels = [f"{s} GW" for s in sizes]
        sizes = [s / bus_size_factor * 1e3 for s in sizes]
        n_cols = 2
    elif specify_buses == "production":
        sizes = [10, 1]
        labels = [f"{s} TWh" for s in sizes]
        sizes = [s / bus_size_factor * 1e6 for s in sizes]
        n_cols = 2
    elif specify_buses == "consumption":
        sizes = [10, 1]
        labels = [f"{s} TWh" for s in sizes]
        sizes = [s / bus_size_factor * 1e6 for s in sizes]
        n_cols = 3

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.23, 1),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = [tech_colors[c] for c in carriers] + [
        color_h2_pipe,
        color_retrofit,
        color_kern,
    ]
    labels = carriers + [
        "H2 pipeline (total)",
        "H2 pipeline (repurposed)",
        "H2 pipeline (Kernnetz)",
    ]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1.13),
        ncol=n_cols,
        frameon=False,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")
    fig.savefig(savepath, bbox_inches="tight")



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

    # add colors for renaming and condensed groups
    for old_name, new_name in carrier_renaming.items():
        if old_name in tech_colors:
            tech_colors[new_name] = tech_colors[old_name]

    for name in c1_groups_name:
        tech_colors[name] = tech_colors[f"urban central {name}"]

    # manual carriers
    tech_colors["urban central oil CHP"] = tech_colors["oil"]

    # Load data
    # _networks = [pypsa.Network(fn) for fn in snakemake.input.networks]
    path = "/home/julian-geis/Documents/04_Ariadne/run_results/20241002_fix_post_discretization_constraints/KN2045_Bal_v4/postnetworks/"
    _networks = [pypsa.Network(fn) for fn in [path + f"elec_s_49_lvopt__none_{i}.nc" for i in range(2020, 2050,5)]]
    modelyears = [fn[-7:-3] for fn in snakemake.input.networks]
    # Hack the transmission projects
    networks = [
        hack_transmission_projects(n.copy(), int(my))
        for n, my in zip(_networks, modelyears)
    ]

    for year in planning_horizons:
        network = networks[planning_horizons.index(year)].copy()

        # electricity supply and demand
        plot_nodal_balance(
            network=network,
            tech_colors=tech_colors,
            start_date="2019-01-01 00:00:00",
            end_date="2019-12-31 00:00:00",
            savepath=f"{snakemake.output.nodal_balances}/elec-all-year-D_{year}.png",
            model_run=snakemake.wildcards.run,
            resample="D",
            plot_lmps=False,
            plot_loads=False,
            nice_names=True,
        )

        plot_nodal_balance(
            network=network,
            tech_colors=tech_colors,
            start_date="2019-01-01 00:00:00",
            end_date="2019-01-31 00:00:00",
            savepath=f"{snakemake.output.nodal_balances}/elec-Jan_{year}.png",
            model_run=snakemake.wildcards.run,
            nice_names=True,
        )

        plot_nodal_balance(
            network=network,
            tech_colors=tech_colors,
            start_date="2019-05-01 00:00:00",
            end_date="2019-05-31 00:00:00",
            savepath=f"{snakemake.output.nodal_balances}/elec-May_{year}.png",
            model_run=snakemake.wildcards.run,
            nice_names=True,
        )

        # storage

        plot_storage(
            network=network,
            tech_colors=tech_colors,
            start_date="2019-01-01 00:00:00",
            end_date="2019-12-31 00:00:00",
            savepath=f"{snakemake.output.results}/storage_{year}.png",
            model_run=snakemake.wildcards.run,
        )

    ## price duration

    networks_dict = {int(my): n for n, my in zip(networks, modelyears)}

    plot_price_duration_curve(
        networks=networks_dict,
        year_colors=year_colors,
        savepath=snakemake.output.elec_price_duration_curve,
        model_run=snakemake.wildcards.run,
        years=planning_horizons,
    )

    plot_price_duration_hist(
        networks=networks_dict,
        year_colors=year_colors,
        savepath=snakemake.output.elec_price_duration_hist,
        model_run=snakemake.wildcards.run,
        years=planning_horizons,
    )

    # make sure to have right file
    fn = "/home/julian-geis/repos/pypsa-ariadne-2/resources/20240920H2KernnetzUpdateHighRes/KN2045_Bal_v4/regions_onshore_elec_s_49.geojson"
    regions = gpd.read_file(fn).set_index("name")

    map_opts = snakemake.params.plotting["map"]
    proj = load_projection(snakemake.params.plotting)

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]
    
    year = 2045
    network = networks[planning_horizons.index(year)].copy()
    plot_h2_map(network, 
                regions,
                savepath=f"{snakemake.output.h2_transmission}_{year}.png")

    regions_de = regions[regions.index.str.startswith("DE")]
    for sb in ["production", "consumption"]:
        network = networks[planning_horizons.index(year)].copy()
        plot_h2_map_de(network, 
                    regions_de,
                    tech_colors = tech_colors,
                    specify_buses=sb,
                    savepath=f"{snakemake.output.h2_transmission}_DE_{sb}_{year}.png",
                    )

    
    
    # ## nodal balances general (Fabians code)
    # dir = snakemake.output.nodal_balances
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    
    # plt.style.use(["bmh", snakemake.input.rc])
    # n = networks[5].copy()

    # months = pd.date_range(freq="M", **snakemake.config["snapshots"]).format(
    #     formatter=lambda x: x.strftime("%Y-%m")
    # )

    # balance = n.statistics.energy_balance(aggregate_time=False)

    # # only DE 
    # ct = "DE"
    # buses = n.buses.index[(n.buses.index.str[:2] == ct)].drop("DE")
    # balance = n.statistics.energy_balance(
    #     aggregate_time=False,
    #     groupby=n.statistics.groupers.get_bus_and_carrier_and_bus_carrier
    #     ).loc[:,buses,:,:].droplevel("bus")


    # n.carriers.color.update(snakemake.config["plotting"]["tech_colors"])
    # colors = n.carriers.color.rename(n.carriers.nice_name)
    # # replace empty values TODO add empty values with colors to plotting config
    # colors[colors.values == ""] = "lightgrey"

    # # wrap in function for multiprocessing
    # def process_group(group, carriers, balance, months, colors):
    #     if not isinstance(carriers, list):
    #         carriers = [carriers]

    #     mask = balance.index.get_level_values("bus_carrier").isin(carriers)
    #     df = balance[mask].groupby("carrier").sum().div(1e3).T

    #     # daily resolution for each carrier
    #     plot_energy_balance_timeseries(
    #         df,
    #         resample="D",
    #         ylabel=group,
    #         colors=colors,
    #         threshold=THRESHOLD,
    #         dir=dir,
    #     )

    #     # monthly resolution for each carrier
    #     plot_energy_balance_timeseries(
    #         df,
    #         resample="M",
    #         ylabel=group,
    #         colors=colors,
    #         threshold=THRESHOLD,
    #         dir=dir,
    #     )

    #     # native resolution for each month and carrier
    #     for month in months:
    #         plot_energy_balance_timeseries(
    #             df,
    #             time=month,
    #             ylabel=group,
    #             colors=colors,
    #             threshold=THRESHOLD,
    #             dir=dir,
    #         )

    # args = [
    #     (group, carriers, balance, months, colors)
    #     for group, carriers in CARRIER_GROUPS.items()
    # ]
    # with Pool(processes=snakemake.threads) as pool:
    #     pool.starmap(process_group, args)

    # # specific cases


    # hydrogen transmission
    # plot whole network: differentiate between Kernnetz and endegenously build links


    ## electricity transmission
    # differentiate between forced in lines / links and endegeneous optiised ones
    # lines: forced in ones are ones from Startnetz: only ones with linetype:
    # links TYNPD in index
    # I'd say: zoom in a bit more on DE, then plot on three separate graphs 
    # total Ausbau (versus 2020), Startnetz (exogen forciert), endogen von PyPSA 

    # total Ausbau
    set_scenario_config(snakemake)
    snakemake.params.plotting["projection"] = {
        "name": "EqualEarth"
    }
    proj = load_projection(snakemake.params.plotting)
    # proj = ccrs.EqualEarth()

    m = networks[5].copy()
    m.mremove("Bus",m.buses[m.buses.x == 0].index )
    m.buses.drop(m.buses.index[m.buses.carrier != "AC"], inplace=True)

    m_base = networks[0].copy()

    # storage as cmap on map
    battery_storage = m.stores[m.stores.carrier.isin(["battery"])]
    regions_de["battery"] = (
        battery_storage.rename(index=battery_storage.bus.str.replace(" battery", "").map(m.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e3)
    )  # GWh
    regions_de["battery"] = regions_de["battery"].where(regions_de["battery"] > 0.1)

    # buses
    bus_size_factor = 0.5e6
    carriers = ["onwind", "solar"]
    elec = m.generators[(m.generators.carrier.isin(carriers)) & (m.generators.bus.str.contains("DE"))].index
    bus_sizes = (
        m.generators.loc[elec, "p_nom_opt"].groupby([m.generators.bus, m.generators.carrier]).sum()
        / bus_size_factor
    )

    # lines
    linew_factor = 1e3
    linkw_factor = 0.5e3

    # get Netzausbau
    line_widths = (m.lines.s_nom_opt - m_base.lines.s_nom_min) / linew_factor
    links = m.links[m.links.carrier == "DC"]
    link_widths = ((m.links.p_nom_opt - m_base.links.p_nom) / linkw_factor).loc[links.index]

    regions_de = regions_de.to_crs(proj.proj4_init)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})
    
    m.plot(
        ax=ax,
        margin=0.06,
        bus_sizes=bus_sizes,
        bus_colors=tech_colors,
        line_widths=line_widths,
        line_colors=tech_colors["AC"],
        link_widths=link_widths,
        link_colors=tech_colors["DC"],
    )

    regions_de.plot(
        ax=ax,
        column="battery",
        cmap="Oranges",
        linewidths=0,
        legend=True,
        legend_kwds={
            "label": "Battery Storage [GWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )
    
    # Set geographic extent for Germany
    ax.set_extent([5.5, 15.5, 47, 56], crs=ccrs.PlateCarree())

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        labelspacing=0.8,
        handletextpad=0,
        frameon=True,
        facecolor="white",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=m.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )
    
    # AC
    sizes_ac = [10, 5]
    labels_ac = [f"HVAC ({s} GW)" for s in sizes_ac]
    scale = 1e3 / linew_factor
    sizes_ac = [s * scale for s in sizes_ac]

    # DC
    sizes_dc = [5, 2]
    labels_dc = [f"HVDC ({s} GW)" for s in sizes_dc]
    scale = 1e3 / linkw_factor
    sizes_dc = [s * scale for s in sizes_dc]

    sizes = sizes_ac + sizes_dc
    labels = labels_ac + labels_dc
    colors = [tech_colors["AC"]] * len(sizes_ac) + [tech_colors["DC"]] * len(sizes_dc)

    legend_kw = dict(
        loc=[0.2, 0.9],
        frameon=True,
        labelspacing=0.5,
        handletextpad=1,
        fontsize=13,
        ncol=2,
        facecolor="white",
    )

    add_legend_lines(
        ax, sizes, labels, colors = colors, legend_kw=legend_kw
    )

    colors = [tech_colors[c] for c in carriers] 
    labels = carriers 
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.9),
        ncol=2,
        frameon=True,
        facecolor="white",
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)
    fig.savefig(f"{snakemake.output.results}/elec_total_ausbau.png", bbox_inches="tight")

    ## Startnetz & TYNDP
    ###################################

    set_scenario_config(snakemake)

    linew_factor = 2e3
    linkw_factor = 1e3

    m = networks[5].copy()
    m.mremove("Bus",m.buses[m.buses.x == 0].index )
    m.buses.drop(m.buses.index[m.buses.carrier != "AC"], inplace=True)

    m_base = networks[0].copy()

    startnetz = m.lines[m.lines.build_year != 0].index
    
    tprojs = m.links.loc[
        (m.links.index.str.startswith("DC") | m.links.index.str.startswith("TYNDP"))
        & ~m.links.reversed
    ].index


    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    
    snakemake.params.plotting["projection"] = {
        "name": "EqualEarth"
    }
    proj = load_projection(snakemake.params.plotting)
    # proj = ccrs.EqualEarth()

    # get exogeneous lines and links
    startnetz_widths = m.lines.s_nom_opt.where(
        m.lines.index.isin(startnetz), other=0.0)  / linew_factor
    tynps_widths = m.links.p_nom_opt.where(
        m.links.index.isin(tprojs), other=0.0) / linkw_factor
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})

    m.plot(
        ax=ax,
        margin=0.06,
        line_widths=startnetz_widths,
        line_colors=tech_colors["AC"],
        link_widths=tynps_widths,
        link_colors=tech_colors["DC"],
    )
    
    # Set geographic extent for Germany
    ax.set_extent([5.5, 15.5, 47, 56], crs=ccrs.PlateCarree())  # Germany bounds
    
    # AC
    sizes_ac = [10, 5]
    labels_ac = [f"HVAC ({s} GW)" for s in sizes_ac]
    scale = 1e3 / linew_factor
    sizes_ac = [s * scale for s in sizes_ac]

    # DC
    sizes_dc = [5, 2]
    labels_dc = [f"HVDC ({s} GW)" for s in sizes_dc]
    scale = 1e3 / linkw_factor
    sizes_dc = [s * scale for s in sizes_dc]

    sizes = sizes_ac + sizes_dc
    labels = labels_ac + labels_dc
    colors = [tech_colors["AC"]] * len(sizes_ac) + [tech_colors["DC"]] * len(sizes_dc)

    legend_kw = dict(
        loc=[0.2, 0.9],
        frameon=True,
        labelspacing=0.5,
        handletextpad=1,
        fontsize=13,
        ncol=2,
        facecolor="white",
    )

    add_legend_lines(
        ax, sizes, labels, colors = colors, legend_kw=legend_kw
    )
    fig.savefig(f"{snakemake.output.results}/elec_exogen.png", bbox_inches="tight")

    ### PyPSA endogeneous
    ###################################
    set_scenario_config(snakemake)

    linew_factor = 2e3
    linkw_factor = 1e3

    m = networks[5].copy()
    m.mremove("Bus",m.buses[m.buses.x == 0].index )
    m.buses.drop(m.buses.index[m.buses.carrier != "AC"], inplace=True)

    m_base = networks[0].copy()

    not_startnetz = m.lines[m.lines.build_year == 0].index
    
    tprojs = m.links.loc[
        (m.links.index.str.startswith("DC") | m.links.index.str.startswith("TYNDP"))
        & ~m.links.reversed
    ].index
    not_tprojs = m.links.loc[
        (m.links.carrier == "DC") & ~(m.links.index.isin(tprojs))
        & ~m.links.reversed
    ].index

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    
    snakemake.params.plotting["projection"] = {
        "name": "EqualEarth"
    }
    proj = load_projection(snakemake.params.plotting)
    # proj = ccrs.EqualEarth()

    # get endogeneous Ausbau
    line_widths = ((m.lines.s_nom_opt - m_base.lines.s_nom_min) / linew_factor).loc[not_startnetz]
    link_widths = ((m.links.p_nom_opt - m_base.links.p_nom) / linkw_factor).loc[not_tprojs]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})

    m.plot(
        ax=ax,
        margin=0.06,
        line_widths=line_widths,
        line_colors=tech_colors["AC"],
        link_widths=link_widths,
        link_colors=tech_colors["DC"],
    )
    
    # Set geographic extent for Germany
    ax.set_extent([5.5, 15.5, 47, 56], crs=ccrs.PlateCarree())  # Germany bounds
    
    # AC
    sizes_ac = [10, 5]
    labels_ac = [f"HVAC ({s} GW)" for s in sizes_ac]
    scale = 1e3 / linew_factor
    sizes_ac = [s * scale for s in sizes_ac]

    # DC
    sizes_dc = [5, 2]
    labels_dc = [f"HVDC ({s} GW)" for s in sizes_dc]
    scale = 1e3 / linkw_factor
    sizes_dc = [s * scale for s in sizes_dc]

    sizes = sizes_ac + sizes_dc
    labels = labels_ac + labels_dc
    colors = [tech_colors["AC"]] * len(sizes_ac) + [tech_colors["DC"]] * len(sizes_dc)

    legend_kw = dict(
        loc=[0.2, 0.9],
        frameon=True,
        labelspacing=0.5,
        handletextpad=1,
        fontsize=13,
        ncol=2,
        facecolor="white",
    )

    add_legend_lines(
        ax, sizes, labels, colors = colors, legend_kw=legend_kw
    )
    fig.savefig(f"{snakemake.output.results}/elec_endogen.png", bbox_inches="tight")




