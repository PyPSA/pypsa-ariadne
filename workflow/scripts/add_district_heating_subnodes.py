# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
from random import randint
from time import sleep

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from geopy.geocoders import Nominatim


# Function to encode city names in UTF-8
def encode_utf8(city_name):
    return city_name.encode("utf-8")


def prepare_subnodes(subnodes, regions_onshore, heat_techs, head=40):
    # TODO: Embed I&O in snakemake rule, add potentials, match CHP capacities

    # Keep only n largest district heating networks according to head parameter
    subnodes = subnodes.sort_values(
        by="Wärmeeinspeisung in GWh/a", ascending=False
    ).head(head)

    # Create a Nominatim object
    nominatim = Nominatim(user_agent="cityEncoder")

    subnodes["lat"] = np.nan
    subnodes["lon"] = np.nan
    subnodes["Stadt"] = subnodes["Stadt"].str.split("_").str[0]

    # Drop duplicates if Gelsenkirchen, Kiel, or Flensburg is included and keep the one with higher Wärmeeinspeisung in GWh/a
    subnodes = subnodes.drop_duplicates(subset="Stadt", keep="first")

    # Get the location of all cities in the dataset (Stadt column) and write them to column "location" do it as try ecxept to avoid errors
    for i, row in subnodes.iterrows():
        try:
            location = nominatim.geocode(encode_utf8(row["Stadt"]), country_codes="DE")
            # Extract the latitude and longitude from the location column
            subnodes.at[i, "lat"] = location.latitude
            subnodes.at[i, "lon"] = location.longitude
            sleep_sec = 1
            sleep(randint(1 * 100, sleep_sec * 100) / 100)
        except:
            logger.info(f"Location not found for {row['Stadt']}")
            pass

    # Make a shapely point object from the lat and lon columns
    subnodes["geometry"] = gpd.points_from_xy(subnodes["lon"], subnodes["lat"])
    # Drop rows with missing geometry
    logger.info("Cities without locations are dropped.")
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
            profile=row["cluster"],
        )

        # Adjust loads of cluster buses
        n.loads_t.p_set.loc[:, f'{row["cluster"]} urban central heat'] *= 1 - scalar

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
            "add_district_heating_subnodes",
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
    heat_techs = gpd.read_file(snakemake.input.heating_technologies_nuts3).set_index(
        "index"
    )
    fernwaermeatlas = pd.read_excel(
        snakemake.input.fernwaermeatlas,
        sheet_name="Fernwärmeatlas_öffentlich",
    )
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    # Assign onshore region to heat techs based on geometry
    heat_techs["cluster"] = heat_techs.apply(
        lambda x: regions_onshore.geometry.contains(x.geometry).idxmax(),
        axis=1,
    )

    subnodes = prepare_subnodes(
        fernwaermeatlas,
        regions_onshore,
        heat_techs,
        head=snakemake.params.district_heating["add_subnodes"],
    )
    subnodes.to_file(snakemake.output.district_heating_subnodes, driver="GeoJSON")

    add_subnodes(n, subnodes)

    n.export_to_netcdf(snakemake.output.network)
