import logging

logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import json
import pypsa


def load_egon():
    """
    Load and prepares the egon data about district heating in Germany on NUTS3 level.

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

    df = pd.read_json(snakemake.input.fn)
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


def update_urban_loads(egon_gdf, n_pre):
    """
    Update district heating demands of clusters according to shares in egon data on NUTS3 level for Germany.
    Other heat loads are adjusted accordingly to ensure consistency of the nodal heat demand.
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
    nodal_uc_shares = nodal_uc_demand / nodal_heat_demand

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


def prepare_subnodes(egon_gdf, head=40):
    # TODO: Embed I&O in snakemake rule, add potentials, match CHP capacities

    # Load and prepare Triebs data
    dh_areas_triebs = pd.read_excel(
        snakemake.input.triebs,
        sheet_name="Staedte",
    )
    # convert dataframe dh_areas_triebs to geopandas dataframe using the Latirude and Longitude columns for geometry column as point coordinates
    dh_areas_triebs["geometry"] = gpd.points_from_xy(
        dh_areas_triebs["Longitude"], dh_areas_triebs["Latitude"]
    )
    dh_areas_triebs = gpd.GeoDataFrame(dh_areas_triebs, geometry="geometry")

    # Keep only n largest district heating networks according to head parameter
    to_keep = ["Stadtname", "NUTS3", "Einwohnerzahl [-]", "geometry"]

    dh_areas_triebs = dh_areas_triebs.sort_values(
        by="Einwohnerzahl [-]", ascending=False
    ).head(head)[to_keep]

    # Merge merged_gdf with dh_areas_triebs using the nuts3 id
    merged_gdf = dh_areas_triebs.merge(
        egon_gdf, left_on="NUTS3", right_on="index", how="left"
    )
    # DH systems without matching NUTS3 id the surrounding region is assigned using the geometries
    merged_gdf.loc[merged_gdf.geometry_y.isna(), "geometry_y"] = merged_gdf.loc[
        merged_gdf.geometry_y.isna()
    ].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_x), "geometry"
        ].item(),
        axis=1,
    )
    merged_gdf.loc[merged_gdf["index"].isna(), "index"] = merged_gdf.loc[
        merged_gdf["index"].isna()
    ].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_x), "index"
        ].item(),
        axis=1,
    )
    merged_gdf.loc[merged_gdf["cluster"].isna(), "cluster"] = merged_gdf.loc[
        merged_gdf["cluster"].isna()
    ].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_x), "cluster"
        ].item(),
        axis=1,
    )
    merged_gdf.loc[
        merged_gdf["District heating"].isna(), "District heating"
    ] = merged_gdf.loc[merged_gdf["District heating"].isna()].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_x), "District heating"
        ].item(),
        axis=1,
    )

    # Intraregional distribution key according to population for NUTS3 regions with multiple DH systems
    merged_gdf["intrareg_dist_key"] = merged_gdf.apply(
        lambda reg: reg["Einwohnerzahl [-]"]
        / merged_gdf.loc[
            merged_gdf["index"] == reg["index"], "Einwohnerzahl [-]"
        ].sum(),
        axis=1,
    ).sort_values()
    # Multiply column District heating with distribution key
    merged_gdf["dh_hh_subnode"] = (
        merged_gdf["District heating"] * merged_gdf["intrareg_dist_key"]
    )

    # Convert district heating counts to percentage of cluster demand

    merged_gdf["demand_share_subnode"] = merged_gdf.apply(
        lambda x: x["dh_hh_subnode"]
        / egon_gdf.loc[egon_gdf.cluster == x.cluster, "District heating"].sum(),
        axis=1,
    )

    # TODO: Function to assign CHPs to subnodes
    return merged_gdf


def add_subnodes(n, subnodes, head=40):
    """
    Add subnodes to the network and adjust loads and capacities accordingly.
    """

    n_sub = n.copy()

    # Add subnodes to network
    for idx, row in subnodes.iterrows():
        name = f'{row["cluster"]} {row["Stadtname"]} urban central heat'

        # Add buses
        n.madd(
            "Bus",
            [name],
            y=row.geometry_x.y,
            x=row.geometry_x.x,
            country="DE",
            location=row["cluster"],
            carrier="urban central heat",
            unit="MWh_th",
        )

        # Add heat loads
        heat_load = row["demand_share_subnode"] * n.loads_t.p_set.filter(
            regex=f"{row['cluster']} urban central heat"
        ).rename(
            {
                f"{row['cluster']} urban central heat": f"{row['cluster']} {row['Stadtname']} urban central heat"
            },
            axis=1,
        )
        n.madd(
            "Load",
            [name],
            bus=name,
            p_set=heat_load,
            carrier="urban central heat",
            location=row["cluster"],
            profile=row["cluster"],
        )

    # Adjust loads of cluster buses
    for idx, row in (
        subnodes.groupby("cluster", as_index=False).sum(numeric_only=True).iterrows()
    ):
        n.loads_t.p_set.loc[:, f'{row["cluster"]} urban central heat'] *= (
            1 - row["demand_share_subnode"]
        )

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
            clusters=22,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Bal_v4",
        )
    logger.info("Adding SysGF-specific functionality")

    n = pypsa.Network(snakemake.input.network)
    egon_gdf = load_egon()
    update_urban_loads(egon_gdf, n)

    if snakemake.params.add_subnodes_de:
        subnodes = prepare_subnodes(egon_gdf, snakemake.params.add_subnodes_de)
        add_subnodes(n, subnodes)
