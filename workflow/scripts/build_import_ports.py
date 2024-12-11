# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build import locations for different carriers by ship for the non-European import options.
"""

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from _helpers import configure_logging, set_scenario_config
from cluster_gas_network import load_bus_regions
from scipy import spatial
from scipy.sparse import csgraph
from shapely.geometry import LineString, Point, Polygon

logger = logging.getLogger(__name__)


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the polygon
    `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    points = np.asarray(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = spatial.Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            with np.errstate(invalid="ignore"):
                poly = poly.intersection(outline)

            polygons.append(poly)

    return polygons


def read_scigrid_gas(fn):
    df = gpd.read_file(fn)
    expanded_param = df.param.apply(json.loads).apply(pd.Series)
    df = pd.concat([df, expanded_param], axis=1)
    df.drop(["param", "uncertainty", "method"], axis=1, inplace=True)
    return df


def build_gem_lng_data(fn):
    df = pd.read_excel(fn, sheet_name="LNG terminals - data", engine="openpyxl")
    df = df.set_index("ComboID")

    remove_country = ["Cyprus", "Turkey"]  # noqa: F841
    remove_terminal = [  # noqa: F841
        "Puerto de la Luz LNG Terminal",
        "Gran Canaria LNG Terminal",
    ]

    status_list = ["Operating", "Construction"]  # noqa: F841

    df = df.query(
        "Status in @status_list \
              & FacilityType == 'Import' \
              & Country != @remove_country \
              & TerminalName != @remove_terminal \
              & CapacityInMtpa != '--' \
              & CapacityInMtpa != 0"
    )

    geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def build_gem_prod_data(fn):
    df = pd.read_excel(fn, sheet_name="Gas extraction - main", engine="openpyxl")
    df = df.set_index("GEM Unit ID")

    remove_country = ["Cyprus", "Türkiye"]  # noqa: F841
    remove_fuel_type = ["oil"]  # noqa: F841

    status_list = ["operating", "in development"]  # noqa: F841

    df = df.query(
        "Status in @status_list \
              & 'Fuel type' != 'oil' \
              & Country != @remove_country \
              & ~Latitude.isna() \
              & ~Longitude.isna()"
    ).copy()

    p = pd.read_excel(fn, sheet_name="Gas extraction - production", engine="openpyxl")
    p = p.set_index("GEM Unit ID")
    p = p[p["Fuel description"].str.contains("gas")]

    capacities = pd.DataFrame(index=df.index)
    for key in ["production", "production design capacity", "reserves"]:
        cap = (
            p.loc[p["Production/reserves"] == key, "Quantity (converted)"]
            .groupby("GEM Unit ID")
            .sum()
            .reindex(df.index)
        )
        # assume capacity such that 3% of reserves can be extracted per year (25% quantile)
        annualization_factor = 0.03 if key == "reserves" else 1.0
        capacities[key] = cap * annualization_factor

    df["mcm_per_year"] = (
        capacities["production"]
        .combine_first(capacities["production design capacity"])
        .combine_first(capacities["reserves"])
    )

    geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def build_gas_input_locations(gem_fn, entry_fn, sto_fn, countries):
    # LNG terminals
    lng = build_gem_lng_data(gem_fn)

    # Entry points from outside the model scope
    entry = read_scigrid_gas(entry_fn)
    entry["from_country"] = entry.from_country.str.rstrip()
    entry = entry.loc[
        ~(entry.from_country.isin(countries) & entry.to_country.isin(countries))
        & ~entry.name.str.contains("Tegelen")  # only take non-EU entries
        | (entry.from_country == "NO")  # malformed datapoint  # entries from NO to GB
    ].copy()

    sto = read_scigrid_gas(sto_fn)
    remove_country = ["RU", "UA", "TR", "BY"]  # noqa: F841
    sto = sto.query("country_code not in @remove_country").copy()

    # production sites inside the model scope
    prod = build_gem_prod_data(gem_fn)

    mcm_per_day_to_mw = 437.5  # MCM/day to MWh/h
    mcm_per_year_to_mw = 1.199  #  MCM/year to MWh/h
    mtpa_to_mw = 1649.224  # mtpa to MWh/h
    mcm_to_gwh = 11.36  # MCM to GWh
    lng["capacity"] = lng["CapacityInMtpa"] * mtpa_to_mw
    entry["capacity"] = entry["max_cap_from_to_M_m3_per_d"] * mcm_per_day_to_mw
    prod["capacity"] = prod["mcm_per_year"] * mcm_per_year_to_mw
    sto["capacity"] = sto["max_cushionGas_M_m3"] * mcm_to_gwh

    lng["type"] = "lng"
    entry["type"] = "pipeline"
    prod["type"] = "production"
    sto["type"] = "storage"

    sel = ["geometry", "capacity", "type"]

    return pd.concat([prod[sel], entry[sel], lng[sel], sto[sel]], ignore_index=True)


def assign_reference_import_sites(gas_input_locations, import_sites, europe_shape):
    europe_shape = europe_shape.squeeze().geometry.buffer(1)  # 1 latlon degree

    for kind in ["lng", "pipeline"]:
        locs = import_sites.query("type == @kind")

        partition = voronoi_partition_pts(locs[["x", "y"]].values, europe_shape)
        partition = gpd.GeoDataFrame(dict(name=locs.index, geometry=partition))
        partition = partition.set_crs(4326).set_index("name")

        match = gpd.sjoin(
            gas_input_locations.query("type == @kind"), partition, how="left"
        )
        gas_input_locations.loc[gas_input_locations["type"] == kind, "port"] = match[
            "name"
        ]

    return gas_input_locations


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_import_ports",
            simpl="",
            clusters="27",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    regions = load_bus_regions(
        snakemake.input.regions_onshore, snakemake.input.regions_offshore
    )

    europe_shape = gpd.read_file(snakemake.input.europe_shape)
    import_sites = pd.read_csv(snakemake.input.reference_import_sites, index_col=0)

    # add a buffer to eastern countries because some
    # entry points are still in Russian or Ukrainian territory.
    buffer = 9000  # meters
    eastern_countries = ["FI", "EE", "LT", "LV", "PL", "SK", "HU", "RO"]
    add_buffer_b = regions.index.str[:2].isin(eastern_countries)
    regions.loc[add_buffer_b] = (
        regions[add_buffer_b].to_crs(3035).buffer(buffer).to_crs(4326)
    )

    countries = regions.index.str[:2].unique().str.replace("GB", "UK")

    gas_input_locations = build_gas_input_locations(
        snakemake.input.gem,
        snakemake.input.entry,
        snakemake.input.storage,
        countries,
    )

    gas_input_locations = assign_reference_import_sites(
        gas_input_locations, import_sites, europe_shape
    )

    gas_input_nodes = gpd.sjoin(gas_input_locations, regions, how="left")

    gas_input_nodes.rename(columns={"name": "bus"}, inplace=True)

    ensure_columns = ["lng", "pipeline", "production", "storage"]
    gas_input_nodes_s = (
        gas_input_nodes.groupby(["bus", "type"])["capacity"]
        .sum()
        .unstack()
        .reindex(columns=ensure_columns)
    )
    gas_input_nodes_s.columns.name = "capacity"
    gas_input_nodes_s.to_csv(snakemake.output.gas_input_nodes_simplified)

    ports = (
        gas_input_nodes.groupby(["bus", "type"])["port"]
        .first()
        .unstack()
        .drop("production", axis=1)
    )
    ports.columns.name = "port"

    ports.to_csv(snakemake.output.ports)
