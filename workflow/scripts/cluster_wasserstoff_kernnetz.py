# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Cluster Wasserstoff Kernnetz to clustered model regions.
"""

import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import pandas as pd
from packaging.version import Version, parse
from pypsa.geo import haversine_pts
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj

def concat_gdf(gdf_list, crs="EPSG:4326"):
    """
    Concatenate multiple geopandas dataframes with common coordinate reference
    system (crs).
    """
    return gpd.GeoDataFrame(pd.concat(gdf_list), crs=crs)

def load_bus_regions(onshore_path, offshore_path):
    """
    Load pypsa-eur on- and offshore regions and concat.
    """
    bus_regions_offshore = gpd.read_file(offshore_path)
    bus_regions_onshore = gpd.read_file(onshore_path)
    bus_regions = concat_gdf([bus_regions_offshore, bus_regions_onshore])
    bus_regions = bus_regions.dissolve(by="name", aggfunc="sum")

    return bus_regions

def build_clustered_h2_network(df, bus_regions, length_factor=1.25):
    for i in [0, 1]:
        gdf = gpd.GeoDataFrame(geometry=df[f"point{i}"], crs="EPSG:4326")

        kws = (
            dict(op="within")
            if parse(gpd.__version__) < Version("0.10")
            else dict(predicate="within")
        )
        bus_mapping = gpd.sjoin(gdf, bus_regions, how="left", **kws).index_right
        bus_mapping = bus_mapping.groupby(bus_mapping.index).first()

        df[f"bus{i}"] = bus_mapping

        df[f"point{i}"] = df[f"bus{i}"].map(
            bus_regions.to_crs(3035).centroid.to_crs(4326)
        )

    # drop pipes where not both buses are inside regions
    df = df.loc[~df.bus0.isna() & ~df.bus1.isna()]

    # drop pipes within the same region
    df = df.loc[df.bus1 != df.bus0]

    # recalculate lengths as center to center * length factor
    df["length"] = df.apply(
        lambda p: length_factor
        * haversine_pts([p.point0.x, p.point0.y], [p.point1.x, p.point1.y]),
        axis=1,
    )

    # tidy and create new numbered index
    df.drop(["point0", "point1"], axis=1, inplace=True)
    df[["bus0", "bus1"]] = df.apply(sort_buses, axis=1)
    df.reset_index(drop=True, inplace=True)

    return df

def sort_buses(row):
    if ((row['bus0'][:2] == row['bus1'][:2]) and (row['bus0'][-1] > row['bus1'][-1])):
        return pd.Series([row['bus1'], row['bus0']])
    elif ((row['bus0'][:2] != row['bus1'][:2]) and (row['bus0'][:2] > row['bus1'][:2])):
        return pd.Series([row['bus1'], row['bus0']])
    else:
        return pd.Series([row['bus0'], row['bus1']])
    
def split_line_by_length(line, segment_length_km):
    """
    Split a Shapely LineString into segments of a specified length.

    Parameters:
    - line (LineString): The original LineString to be split.
    - segment_length_km (float): The desired length of each resulting segment in kilometers.

    Returns:
    list: A list of Shapely LineString objects representing the segments.
    """
    # Define a function for projecting points to meters
    project_to_meters = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'),  # assuming WGS84
        pyproj.Proj(proj='utm', zone=33, ellps='WGS84'),  # adjust the projection as needed
        always_xy=True
    ).transform

    # Define a function for projecting points back to decimal degrees
    project_to_degrees = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='utm', zone=33, ellps='WGS84'),  # adjust the projection as needed
        pyproj.Proj('epsg:4326'),
        always_xy=True
    ).transform

    # Convert segment length from kilometers to meters
    segment_length_meters = segment_length_km * 1000.0

    # Project the LineString to a suitable metric projection
    projected_line = transform(project_to_meters, line)

    total_length = projected_line.length
    num_segments = int(total_length / segment_length_meters)

    if num_segments < 2:
        return [line]

    segments = []
    for i in range(1, num_segments + 1):
        start_point = projected_line.interpolate((i - 1) * segment_length_meters)
        end_point = projected_line.interpolate(i * segment_length_meters)

        # Extract x and y coordinates from the tuples
        start_point_coords = (start_point.x, start_point.y)
        end_point_coords = (end_point.x, end_point.y)

        # Create Shapely Point objects
        start_point_degrees = Point(start_point_coords)
        end_point_degrees = Point(end_point_coords)

        # Project the points back to decimal degrees
        start_point_degrees = transform(project_to_degrees, start_point_degrees)
        end_point_degrees = transform(project_to_degrees, end_point_degrees)

        # last point without interpolation
        if i == num_segments:
            end_point_degrees = Point(line.coords[-1])


        segment = LineString([start_point_degrees, end_point_degrees])
        segments.append(segment)

    return segments

def divide_pipes(df, segment_length=10):
    """
    Divide a GeoPandas DataFrame of LineString geometries into segments of a specified length.

    Parameters:
    - df (GeoDataFrame): The input DataFrame containing LineString geometries.
    - segment_length (float): The desired length of each resulting segment in kilometers.

    Returns:
    GeoDataFrame: A new GeoDataFrame with additional rows representing the segmented pipes.
    """

    result = pd.DataFrame(columns=df.columns)

    for index, pipe in df.iterrows():
        segments = split_line_by_length(pipe.geometry, segment_length)

        for i in range(0,len(segments)):
            res_row = pipe.copy()
            res_row.geometry = segments[i]
            res_row.point0 = Point(segments[i].coords[0])
            res_row.point1 = Point(segments[i].coords[1])
            res_row.length_haversine = segment_length
            result.loc[f"{index}-{i}"] = res_row

    return result


def reindex_pipes(df):
    def make_index(x):
        connector = " <-> " if x.bidirectional else " -> "
        return "h2 pipeline " + x.bus0 + connector + x.bus1

    df.index = df.apply(make_index, axis=1)

    df["p_min_pu"] = df.bidirectional.apply(lambda bi: -1 if bi else 0)
    df.drop("bidirectional", axis=1, inplace=True)

    df.sort_index(axis=1, inplace=True)


def aggregate_parallel_pipes(df):
    strategies = {
        "bus0": "first",
        "bus1": "first",
        "p_nom": "sum",
        "max_pressure_bar": "mean",
        "build_year": "mean",
        "diameter_mm": "mean",
        "length": "mean",
        "name": " ".join,
        "p_min_pu": "min",
    }
    return df.groupby(df.index).agg(strategies)


if __name__ == "__main__":
    '''
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_wasserstoff_kernnetz", simpl="", clusters="37")
    '''

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    fn = snakemake.input.cleaned_h2_network
    df = pd.read_csv(fn, index_col=0)
    for col in ["point0", "point1", "geometry"]:
        df[col] = df[col].apply(wkt.loads)

    bus_regions = load_bus_regions(
        snakemake.input.regions_onshore, snakemake.input.regions_offshore
    )

    if snakemake.config["wasserstoff_kernnetz"]["divide_pipes"]:
        df = divide_pipes(df, segment_length=snakemake.config["wasserstoff_kernnetz"]["pipes_segment_length"])

    wasserstoff_kernnetz = build_clustered_h2_network(df, bus_regions)

    reindex_pipes(wasserstoff_kernnetz)

    wasserstoff_kernnetz = aggregate_parallel_pipes(wasserstoff_kernnetz)

    wasserstoff_kernnetz.to_csv(snakemake.output.clustered_h2_network)