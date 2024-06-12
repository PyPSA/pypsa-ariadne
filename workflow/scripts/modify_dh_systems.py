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
        71: "District heating",
    }

    df = pd.read_json(snakemake.input.fn)
    id_region = pd.read_json(snakemake.input.fn_map)

    df["internal_id"] = df["internal_id"].apply(lambda x: x[0])
    df = df[df["internal_id"] == 71]  # Keep only rows with district heating

    df["nuts3"] = df.id_region.map(
        id_region.set_index(id_region.id_region_from).kuerzel_to
    )

    heat_tech_per_region = df.groupby([df.nuts3, df.internal_id]).sum().value.unstack()
    heat_tech_per_region.rename(columns=internal_id, inplace=True)

    egon_df = heat_tech_per_region.merge(nuts3, left_on="nuts3", right_on="index")
    egon_gdf = gpd.GeoDataFrame(egon_df)  # Convert merged DataFrame to GeoDataFrame

    return egon_gdf


def update_dist_shares(egon_gdf, n_pre):
    """
    Update district heating shares of clusters according to egon data on NUTS3 level.
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

    # Calculate DH demand shares
    dh_shares = (
        egon_gdf.groupby("cluster")["District heating"].sum()
        / egon_gdf["District heating"].sum()
    )

    # Current DH demands by cluster
    dh_demand = n.loads_t.p_set.filter(regex="DE\d.*urban central heat")

    return n


def prepare_subnodes_de(egon_gdf):
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

    # Merge merged_gdf with dh_areas_triebs using the nuts3 id
    merged_gdf = egon_gdf.merge(
        dh_areas_triebs, left_on="index", right_on="NUTS3", how="right"
    )
    # Create additional column nuts_3_matchshape that contains the value of the index column in merged_gdf of the row where the geometry column of dh_areas_triebs intersects with the geometry column of nuts3_shapes

    merged_gdf.loc[merged_gdf.geometry_x.isna(), "geometry_x"] = merged_gdf.loc[
        merged_gdf.geometry_x.isna()
    ].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_y), "geometry"
        ].item(),
        axis=1,
    )
    merged_gdf.loc[merged_gdf["index"].isna(), "index"] = merged_gdf.loc[
        merged_gdf["index"].isna()
    ].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_y), "index"
        ].item(),
        axis=1,
    )
    merged_gdf.loc[
        merged_gdf["District heating"].isna(), "District heating"
    ] = merged_gdf.loc[merged_gdf["District heating"].isna()].apply(
        lambda x: egon_gdf.loc[
            egon_gdf.geometry.contains(x.geometry_y), "District heating"
        ].item(),
        axis=1,
    )

    # Intraregional distribution key according to population
    merged_gdf["intrareg_dist_key"] = merged_gdf.apply(
        lambda reg: reg["Einwohnerzahl [-]"]
        / merged_gdf.loc[
            merged_gdf["index"] == reg["index"], "Einwohnerzahl [-]"
        ].sum(),
        axis=1,
    ).sort_values()
    # Multiply column District heating with distribution key
    merged_gdf["demand_dh_subnode"] = (
        merged_gdf["District heating"] * merged_gdf["intrareg_dist_key"]
    )

    return merged_gdf


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
    update_dist_shares(egon_gdf, n)

    if snakemake.params.enable_subnodes_de:
        prepare_subnodes_de(egon_gdf)
