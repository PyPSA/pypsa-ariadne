# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from shapely.geometry import Point


def load_egon():
    """
    Load and prepares the egon data about district heating in Germany on NUTS3
    level.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the processed egon data.
    """

    nuts3 = gpd.read_file(snakemake.input.nuts3)[
        ["index", "pop", "geometry"]
    ]  # Keep only necessary columns

    internal_id = {
        9: "Hard coal",
        10: "Brown coal",
        11: "Natural gas",
        34: "Heating oil",
        35: "Biomass (solid)",
        68: "Ambient heating",
        69: "Solar heat",
        71: "District heating",
        72: "Electrical energy",
        218: "Biomass (excluding wood, biogas)",
    }

    # Load Json from snakemakeinput and write into dataframe df

    with open(snakemake.input.fn) as datafile:
        data = json.load(datafile)["data"]
    df = pd.DataFrame(data)

    id_region = pd.read_json(snakemake.input.fn_map)

    df["internal_id"] = df["internal_id"].apply(lambda x: x[0])
    # df = df[df["internal_id"] == 71]  # Keep only rows with district heating

    df["nuts3"] = df.id_region.map(
        id_region.set_index(id_region.id_region_from).kuerzel_to
    )

    heat_tech_per_region = df.groupby([df.nuts3, df.internal_id]).sum().value.unstack()
    heat_tech_per_region.rename(columns=internal_id, inplace=True)

    egon_df = heat_tech_per_region.merge(nuts3, left_on="nuts3", right_on="index")
    egon_gdf = gpd.GeoDataFrame(egon_df)  # Convert merged DataFrame to GeoDataFrame
    egon_gdf = egon_gdf.to_crs("EPSG:4326")

    return egon_gdf


def update_urban_loads_de(egon_gdf, n_pre):
    """
    Update district heating demands of clusters according to shares in egon
    data on NUTS3 level for Germany.

    Other heat loads are adjusted accodingly to ensure consistency of
    the nodal heat demand.
    """

    n = n_pre.copy()
    regions_onshore = gpd.read_file(
        snakemake.input.regions_onshore
    )  # shared resources true
    regions_onshore.set_index("name", inplace=True)
    # Map NUTS3 regions of egon data to corresponding clusters according to maximum overlap

    egon_gdf["cluster"] = egon_gdf.apply(
        lambda x: regions_onshore.geometry.intersection(x.geometry).area.idxmax(),
        axis=1,
    )

    # Calculate nodal DH shares according to households and modify index
    egon_gdf_clustered = egon_gdf.groupby("cluster").sum(numeric_only=True)
    nodal_dh_shares = egon_gdf_clustered["District heating"] / egon_gdf_clustered.drop(
        "pop", axis=1
    ).sum(axis=1)

    dh_shares = pd.read_csv(snakemake.input.district_heat_share, index_col=0)
    urban_fraction = dh_shares["urban fraction"]
    max_dh_share = snakemake.params.district_heating["potential"]
    progress = snakemake.params.district_heating["progress"][
        int(snakemake.wildcards.planning_horizons)
    ]

    diff = ((urban_fraction * max_dh_share) - nodal_dh_shares).clip(lower=0).dropna()
    nodal_dh_shares += diff * progress

    nodal_dh_shares.index += " urban central heat"

    # District heating demands by cluster in German nodes before heat distribution
    nodal_uc_demand = (
        n.loads_t.p_set.filter(regex="DE.*urban central heat")
        .apply(lambda c: c * n.snapshot_weightings.generators)
        .sum()
        .div(
            1 + snakemake.config["sector"]["district_heating"]["district_heating_loss"]
        )
    )

    nodal_uc_losses = (
        nodal_uc_demand
        - n.loads_t.p_set.filter(regex="DE.*urban central heat")
        .apply(lambda c: c * n.snapshot_weightings.generators)
        .sum()
    )

    # Sum of rural and urban heat demand
    nodal_heat_demand = (
        n.loads_t.p_set.filter(regex="DE.*heat$")
        .apply(lambda c: c * n.snapshot_weightings.generators)
        .sum()
        .sub(nodal_uc_losses, fill_value=0)
    )

    # Modify index of nodal_heat demand to align with urban central loads and aggregate loads
    nodal_heat_demand.index = nodal_heat_demand.index.str.replace(
        "decentral", "central"
    ).str.replace("rural", "urban central")

    nodal_heat_demand = nodal_heat_demand.groupby(nodal_heat_demand.index).sum()

    # Old district heating share
    nodal_uc_shares = (nodal_uc_demand / nodal_heat_demand).fillna(0)

    # Scaling factor for update of urban central heat loads

    scaling_factor = nodal_dh_shares / nodal_uc_shares
    scaling_factor.dropna(
        inplace=True
    )  # To deal with shape anomaly described in https://github.com/PyPSA/pypsa-eur/issues/1100

    # Update urban heat loads changing distribution and restoring old scale
    old_uc_loads = n.loads_t.p_set.filter(regex="DE.*urban central heat")
    new_uc_loads = (
        n.loads_t.p_set.filter(regex="DE.*urban central heat") * scaling_factor
    )
    restore_scalar = new_uc_loads.sum().sum() / old_uc_loads.sum().sum()
    new_uc_loads = new_uc_loads / restore_scalar
    diff_update = new_uc_loads - old_uc_loads
    diff_update.columns = diff_update.columns.str.replace("central", "decentral")

    n_pre.loads_t.p_set[new_uc_loads.columns] = new_uc_loads
    n_pre.loads_t.p_set[diff_update.columns] -= diff_update

    return


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_dh_systems",
            simpl="",
            clusters=44,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Bal_v4",
        )
    logger.info("Adding SysGF-specific functionality")

    n = pypsa.Network(snakemake.input.network)
    egon_gdf = load_egon()
    update_urban_loads_de(egon_gdf, n)

    n.export_to_netcdf(snakemake.output[0])
