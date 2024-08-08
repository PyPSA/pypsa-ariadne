# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def cluster_egon(heat_techs):
    """
    Map NUTS3 regions of egon data to corresponding clusters according to
    maximum overlap.
    """

    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_onshore.set_index("name", inplace=True)
    # Map NUTS3 regions of egon data to corresponding clusters according to maximum overlap

    heat_techs["cluster"] = heat_techs.apply(
        lambda x: regions_onshore.geometry.intersection(x.geometry).area.idxmax(),
        axis=1,
    )

    # Group and aggregate by cluster
    heat_techs_clustered = heat_techs.groupby("cluster").sum(numeric_only=True)

    return heat_techs_clustered


def update_urban_loads_de(heat_techs_clustered):
    """
    Update district heating demands of clusters according to shares in egon
    data on NUTS3 level for Germany.

    Other heat loads are adjusted accodingly to ensure consistency of
    the nodal heat demand.
    """

    nodal_dh_shares = heat_techs_clustered[
        "District heating"
    ] / heat_techs_clustered.drop("pop", axis=1).sum(axis=1)

    dh_shares = pd.read_csv(snakemake.input.district_heat_share, index_col=0)
    urban_fraction = dh_shares["urban fraction"]
    max_dh_share = snakemake.params.district_heating["potential"]
    progress = snakemake.params.district_heating["progress"][
        int(snakemake.wildcards.planning_horizons)
    ]

    diff = ((urban_fraction * max_dh_share) - nodal_dh_shares).clip(lower=0).dropna()
    nodal_dh_shares += diff * progress
    nodal_dh_shares = nodal_dh_shares.filter(regex="DE")
    dh_shares.loc[nodal_dh_shares.index, "district fraction of node"] = nodal_dh_shares
    dh_shares.loc[nodal_dh_shares.index, "urban fraction"] = pd.concat(
        [urban_fraction.loc[nodal_dh_shares.index], nodal_dh_shares], axis=1
    ).max(axis=1)

    return dh_shares


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_district_heat_share",
            simpl="",
            clusters=44,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Bal_v4",
        )
    logger.info("Updating district heating shares with egon data")

    heat_techs = gpd.read_file(snakemake.input.heating_technologies_nuts3)

    heat_techs_clustered = cluster_egon(heat_techs)
    dh_shares = update_urban_loads_de(heat_techs_clustered)

    dh_shares.to_csv(snakemake.output.district_heat_share)
