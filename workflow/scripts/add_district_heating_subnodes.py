# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
from random import randint
from time import sleep

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from geopy.geocoders import Nominatim


# Function to encode city names in UTF-8
def encode_utf8(city_name):
    return city_name.encode("utf-8")


def prepare_subnodes(subnodes, cities, regions_onshore, heat_techs, head=40):
    # TODO: Embed I&O in snakemake rule, add potentials, match CHP capacities
    # If head is boolean set it to 40 for default behavior
    if isinstance(head, bool):
        head = 40

    subnodes["Stadt"] = subnodes["Stadt"].str.split("_").str[0]

    # Drop duplicates if Gelsenkirchen, Kiel, or Flensburg is included and keep the one with higher Wärmeeinspeisung in GWh/a
    subnodes = subnodes.drop_duplicates(subset="Stadt", keep="first")

    # Keep only n largest district heating networks according to head parameter
    subnodes = subnodes.sort_values(
        by="Wärmeeinspeisung in GWh/a", ascending=False
    ).head(head)

    subnodes["geometry"] = subnodes["Stadt"].apply(
        lambda s: cities.loc[cities["Stadt"] == s, "geometry"].values[0]
    )

    subnodes = subnodes.dropna(subset=["geometry"])
    # Convert the DataFrame to a GeoDataFrame
    subnodes = gpd.GeoDataFrame(subnodes, crs="EPSG:4326")

    # Assign cluster to subnodes according to onshore regions
    subnodes["cluster"] = subnodes.apply(
        lambda x: regions_onshore.geometry.contains(x.geometry).idxmax(), axis=1
    )
    subnodes["nuts3"] = subnodes.apply(
        lambda x: heat_techs.geometry.contains(x.geometry).idxmax(),
        axis=1,
    )
    subnodes["nuts3_shape"] = subnodes.apply(
        lambda x: heat_techs.loc[
            heat_techs.geometry.contains(x.geometry).idxmax(), "geometry"
        ].wkt,
        axis=1,
    )

    return subnodes


def add_subnodes(n, subnodes):
    """
    Add subnodes to the network and adjust loads and capacities accordingly.
    """

    # Add subnodes to network
    for idx, row in subnodes.iterrows():
        name = f'{row["cluster"]} {row["Stadt"]} urban central heat'

        # Add buses
        n.madd(
            "Bus",
            [name],
            y=row.geometry.y,
            x=row.geometry.x,
            country="DE",
            location=row["cluster"],
            carrier="urban central heat",
            unit="MWh_th",
        )

        # Add heat loads
        scalar = min(
            1,
            (
                row["Wärmeeinspeisung in GWh/a"]
                * 1e3
                / n.loads_t.p_set.filter(regex=f"{row['cluster']} urban central heat")
                .sum(axis=1)
                .mul(n.snapshot_weightings.generators)
                .sum()
            ),
        )
        lost_load = (
            row["Wärmeeinspeisung in GWh/a"] * 1e3
            - n.loads_t.p_set.filter(regex=f"{row['cluster']} urban central heat")
            .sum(axis=1)
            .mul(n.snapshot_weightings.generators)
            .sum()
        )
        if scalar == 1:
            logger.info(
                f"District heating load of {row['Stadt']} exceeds load of its assigned cluster {row['cluster']}. {lost_load} MWh/a are disregarded."
            )
        heat_load = scalar * n.loads_t.p_set.filter(
            regex=f"{row['cluster']} urban central heat"
        ).rename(
            {
                f"{row['cluster']} urban central heat": f"{row['cluster']} {row['Stadt']} urban central heat"
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
        )

        # Adjust loads of cluster buses
        n.loads_t.p_set.loc[:, f'{row["cluster"]} urban central heat'] *= 1 - scalar

        # Replicate district heating stores and links of mother node for subnodes
        # TODO: Add heat pump links

        n.madd(
            "Bus",
            [f"{row['cluster']} {row['Stadt']} urban central water tanks"],
            location=row["cluster"],
            carrier="urban central water tanks",
            unit="MWh_th",
        )

        stores = (
            n.stores.filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("Store")
        )
        n.madd("Store", stores.index, **stores)

        links = (
            n.links.loc[~n.links.carrier.str.contains("heat pump")]
            .filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("Link")
        )
        n.madd("Link", links.index, **links)

        # Add heat pumps to subnode
        heat_pumps = (
            n.links.filter(regex=f"{row['cluster']} urban central.*heat pump", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("Link")
        ).drop("efficiency", axis=1)
        heat_pumps_t = n.links_t.efficiency.filter(
            regex=f"{row['cluster']} urban central.*heat pump"
        )
        heat_pumps_t.columns = heat_pumps_t.columns.str.replace(
            f"{row['cluster']} urban central",
            f"{row['cluster']} {row['Stadt']} urban central",
        )
        n.madd("Link", heat_pumps.index, efficiency=heat_pumps_t, **heat_pumps)

        # Add artificial gas boiler to subnode
        n.madd(
            "Generator",
            [f"{name} load shedding"],
            bus=name,
            carrier="gas",
            p_nom_extendable=True,
            p_nom_max=1e6,
            capital_cost=10000,
            marginal_cost=200,
            efficiency=1,
        )

    return


def extend_cops(cops, subnodes):
    """
    Extend COPs by subnodes mirroring the timeseries of the corresponding
    mother node.
    """
    cops_extended = cops.copy()

    # Iterate over the DataFrame rows
    for _, row in subnodes.iterrows():
        cluster_name = row["cluster"]
        city_name = row["Stadt"]

        # Select the xarray entry where name matches the cluster
        selected_entry = cops.sel(name=cluster_name)

        # Rename the selected entry
        renamed_entry = selected_entry.assign_coords(name=f"{cluster_name} {city_name}")

        # Combine the renamed entry with the extended dataset
        cops_extended = xr.concat([cops_extended, renamed_entry], dim="name")

    # Change dtype of the name dimension to string
    cops_extended.coords["name"] = cops_extended.coords["name"].astype(str)

    return cops_extended


def extend_heating_distribution(existing_heating_distribution, subnodes):
    """
    Extend heating distribution by subnodes mirroring the distribution of the
    corresponding mother node.
    """
    # Merge the existing heating distribution with subnodes on the cluster name
    mother_nodes = existing_heating_distribution.loc[subnodes.cluster.unique()]
    mother_nodes["cities"] = subnodes.groupby("cluster")["Stadt"].apply(list)
    # Explode the list of cities
    mother_nodes = mother_nodes.explode(("cities", ""))
    mother_nodes.index = mother_nodes.index + " " + mother_nodes[("cities", "")]
    mother_nodes.drop(columns=("cities", ""), inplace=True)
    existing_heating_distribution_extended = pd.concat(
        [existing_heating_distribution, mother_nodes]
    )
    return existing_heating_distribution_extended


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_district_heating_subnodes",
            simpl="",
            clusters=27,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Bal_v4",
        )

    logger.info("Adding SysGF-specific functionality")

    n = pypsa.Network(snakemake.input.network)
    heat_techs = gpd.read_file(snakemake.input.heating_technologies_nuts3).set_index(
        "index"
    )
    fernwaermeatlas = pd.read_excel(
        snakemake.input.fernwaermeatlas,
        sheet_name="Fernwärmeatlas_öffentlich",
    )
    cities = gpd.read_file(snakemake.input.cities)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    # Assign onshore region to heat techs based on geometry
    heat_techs["cluster"] = heat_techs.apply(
        lambda x: regions_onshore.geometry.contains(x.geometry).idxmax(),
        axis=1,
    )

    subnodes = prepare_subnodes(
        fernwaermeatlas,
        cities,
        regions_onshore,
        heat_techs,
        head=snakemake.params.district_heating["add_subnodes"],
    )
    subnodes.to_file(snakemake.output.district_heating_subnodes, driver="GeoJSON")

    add_subnodes(n, subnodes)

    if snakemake.config["foresight"] == "myopic":
        cops = xr.open_dataarray(snakemake.input.cop_profiles)
        cops_extended = extend_cops(cops, subnodes)
        cops_extended.to_netcdf(snakemake.output.cop_profiles_extended)

    if snakemake.wildcards.planning_horizons == str(snakemake.params["baseyear"]):
        existing_heating_distribution = pd.read_csv(
            snakemake.input.existing_heating_distribution,
            header=[0, 1],
            index_col=0,
        )
        existing_heating_distribution_extended = extend_heating_distribution(
            existing_heating_distribution, subnodes
        )
        existing_heating_distribution_extended.to_csv(
            snakemake.output.existing_heating_distribution_extended
        )
    n.export_to_netcdf(snakemake.output.network)
