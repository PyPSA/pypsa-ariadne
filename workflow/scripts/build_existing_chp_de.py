# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Using BNetzA data to get a high resolution map of German CHP plants.
(https://open-mastr.readthedocs.io/en/latest/).
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
import os
import sys
    

def clean_data(combustion, biomass, geodata):
    """
    Clean the data and return a dataframe with the relevant information.
    PLZ is translated to longitude and latitude using the pyGeoDb data.
    """
    biomass.dropna(subset="Postleitzahl", inplace=True)
    biomass.rename(columns={'NameStromerzeugungseinheit': 'NameKraftwerk'}, inplace=True)

    data = pd.concat([biomass, combustion], join='inner', ignore_index=True)

    # Get only CHP plants
    CHP_raw = data.query("ThermischeNutzleistung > 0").copy()
    CHP_raw.NameKraftwerk = CHP_raw.NameKraftwerk.fillna(CHP_raw.EinheitMastrNummer)

    rename_columns = {
        "NameKraftwerk": "Name",
        "Energietraeger": "Fueltype",
        "Technologie": "Technology",
        "ElektrischeKwkLeistung": "Capacity",
        "ThermischeNutzleistung": "Capacity_thermal",
        "Inbetriebnahmedatum": "DateIn",
        "DatumEndgueltigeStilllegung": "DateOut",
        "Postleitzahl": "Postleitzahl",
        "Breitengrad": "lat",
        "Laengengrad": "lon",
    }
    CHP_sel = CHP_raw[rename_columns.keys()].rename(columns=rename_columns)

    # set missing information to match the powerplant data format
    CHP_sel[["Set", "Country", "Efficiency"]] = ["CHP", "DE", ""]

    # change date format
    CHP_sel.DateIn = CHP_sel.DateIn.str[:4].astype(float)
    CHP_sel.DateOut = CHP_sel.DateOut.str[:4].astype(float)

    # get location from PLZ
    CHP_sel.fillna({"lat": CHP_sel.Postleitzahl.map(geodata.lat)}, inplace=True)
    CHP_sel.fillna({"lon": CHP_sel.Postleitzahl.map(geodata.lng)}, inplace=True)

    fueltype = {
        "Erdgas": "Natural Gas",
        "Mineralölprodukte": "Oil",
        "Steinkohle": "Coal",
        "Braunkohle": "Lignite",
        "andere Gase": "Natural Gas",
        "nicht biogenere Abfälle": "Waste",
        "Wärme": "Other",
        "Biomasse": "Bioenergy",
        "Wasserstoff": "Hydrogen",
    }
    technology = {
        "Verbrennungsmotor": "",
        "Gasturbinen mit Abhitzekessel": "CCGT",
        "Brennstoffzelle": "Fuel Cell",
        "Strilingmotor": "",
        'Kondensationsmaschine mit Entnahme': "Steam Turbine", 
        'Sonstige': "",
        'Gasturbinen ohne Abhitzekessel': "OCGT",
        'Dampfmotor': "Steam Turbine",
        'Gegendruckmaschine mit Entnahme': "Steam Turbine",
        'Gegendruckmaschine ohne Entnahme':"Steam Turbine",
        'Gasturbinen mit nachgeschalteter Dampfturbine': "CCGT",
        'ORC (Organic Rankine Cycle)-Anlage': "Steam Turbine",
        'Kondensationsmaschine ohne Entnahme': "Steam Turbine",
    }

    CHP_sel.replace({"Fueltype": fueltype, "Technology": technology}, inplace=True)

    def lookup_geodata(missing_plz):
        for i in range(10):
            plz = missing_plz[:-1] + str(i)
            if plz in geodata.index:
                return geodata.loc[plz]
        for i in range(100):
            prefix = "0" if i < 10 else ""
            plz = missing_plz[:-2] + prefix + str(i)
            if plz in geodata.index:
                return geodata.loc[plz]

        return pd.Series((pd.NA, pd.NA))

    missing_i = CHP_sel.lat.isna() | CHP_sel.lon.isna()
    CHP_sel.loc[missing_i, ["lat", "lon"]] = CHP_sel.loc[missing_i, "Postleitzahl"].apply(lookup_geodata)

    cols = [
        'Name',
        'Fueltype',
        'Technology',
        'Set',
        'Country',
        'Capacity',
        'Efficiency',
        'DateIn',
        'DateOut',
        'lat',
        'lon',
        'Capacity_thermal'
    ]

    return CHP_sel[cols].copy()


def calculate_efficiency(CHP_de):
    """
    Calculate the efficiency of the CHP plants depending on Capacity and DateIn.
    Following Triebs et al. (https://doi.org/10.1016/j.ecmx.2020.100068)
    """
    def EXT(cap, year):
        # returns the efficiency for extraction condensing turbine
        return ((44/2400) * cap + 0.125 * year - 204.75) / 100

    def BP(cap,year):
        # returns the efficiency for back pressure turbine
        return ((5e-3) * cap + 0.325 * year - 611.75) / 100
    # TODO: differentiate between extraction condensing turbine and back pressure turbine
    CHP_de['Efficiency'] = CHP_de.apply(lambda row: BP(row['Capacity'], row['DateIn']), axis=1)
        
    return CHP_de
    

if __name__ == "__main__":
    if "snakemake" not in globals():
        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_existing_chp_de")
    
    logging.basicConfig(level=snakemake.config["logging"]["level"])

    biomass = pd.read_csv(snakemake.input.mastr_biomass, dtype={"Postleitzahl": str})
    combustion = pd.read_csv(snakemake.input.mastr_combustion, dtype={"Postleitzahl": str})
    
    geodata = pd.read_csv(
        snakemake.input.plz_mapping[0],
        index_col="plz",
        dtype={"plz": str},
        names=["plz", "lat", "lng"],
        skiprows=1
    )

    CHP_de = clean_data(combustion, biomass, geodata)

    CHP_de = calculate_efficiency(CHP_de)
    
    CHP_de.to_csv(snakemake.output.german_chp, index=False)