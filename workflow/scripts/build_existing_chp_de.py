# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Using BNetzA data to get a high resolution map of German CHP plants.
(https://open-mastr.readthedocs.io/en/latest/).
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
import requests
import zipfile
import tarfile
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("../.."))

def load_dataset(mastr):
    """
    Load the dataset from the BNetzA website and return it as a dataframe.
    https://zenodo.org/records/8225106
    """
    
    source_file = mastr
    # Specify the destination directory path
    destination_directory = "./data/"
    
    # Move the file to the destination directory
    shutil.move(source_file, destination_directory)
    # Delete the original directory
    shutil.rmtree("./zenodo.org/records/8225106/")
    os.rename("./data/bnetza_open_mastr_2023-08-08_B.zip?download=1", "./data/bnetza_open_mastr_2023-08-08_B.zip")
    # Unpack the zip file
    with zipfile.ZipFile("./data/bnetza_open_mastr_2023-08-08_B.zip", "r") as zip_ref:
        # Extract the specific file you want
        files_to_extract = ["bnetza_open_mastr_2023-08-08_B_biomass.csv", "bnetza_open_mastr_2023-08-08_B_combustion.csv"]
        for file in files_to_extract:
            zip_ref.extract(file, path="./data")
    os.remove("./data/bnetza_open_mastr_2023-08-08_B.zip")
    # Load the data
    combustion = pd.read_csv("./data/bnetza_open_mastr_2023-08-08_B_combustion.csv", dtype={"Postleitzahl": str})
    biomass = pd.read_csv("./data/bnetza_open_mastr_2023-08-08_B_biomass.csv", dtype={"Postleitzahl": str})

    return combustion, biomass

def load_plz_mapping(plz_data):
    """
    Load the plz mapping from pyGeoDb
    https://pypi.org/project/pyGeoDb/#modal-close
    """
    file_path = "./data/plzdata.py"
    if not os.path.exists(file_path):
        source_file = plz_data
        # Specify the destination directory path
        destination_directory = "./data/"
        # Move the file to the destination directory
        shutil.move(source_file, destination_directory)
        # Delete the original directory
        shutil.rmtree("./files.pythonhosted.org")
        # Extract files
        with tarfile.open("./data/pyGeoDb-1.3.tar.gz", 'r:gz') as tar:
            tar.extractall("./data/")
        # read file
        shutil.move("./data/pyGeoDb-1.3/pygeodb/plzdata.py", "./data/")
        # Delete the .tar.gz file
        shutil.rmtree("./data/pyGeoDb-1.3")
        os.remove("./data/pyGeoDb-1.3.tar.gz")
    else:
        print("plzdata.py already exists")

    sys.path.append('./data')
    from plzdata import geodata
    return geodata
    

def clean_data(combustion, biomass, geodata):
    """
    Clean the data and return a dataframe with the relevant information.
    PLZ is translated to longitude and latitude using the pyGeoDb data.
    """
    biomass = biomass[(biomass['Postleitzahl'] != 0) & biomass['Postleitzahl'].notnull()]
    biomass.rename(columns={'NameStromerzeugungseinheit': 'NameKraftwerk'}, inplace=True)

    data = pd.concat([biomass, combustion], axis=0, join='inner', ignore_index=True)

    # Get only CHP plants
    CHP_raw = data[(data["ThermischeNutzleistung"] > 0) & (data["ThermischeNutzleistung"].notnull())]
    CHP_raw.loc[:, "NameKraftwerk"] = CHP_raw["NameKraftwerk"].fillna(CHP_raw["EinheitMastrNummer"])

    CHP_sel = CHP_raw[["NameKraftwerk", 
                   "Energietraeger", 
                   "Technologie", 
                   "Postleitzahl", 
                   "Inbetriebnahmedatum", 
                   "DatumEndgueltigeStilllegung", 
                   "ThermischeNutzleistung", 
                   "ElektrischeKwkLeistung",
                   ]]
    rename_columns = {
    "NameKraftwerk": "Name",
    "Energietraeger": "Fueltype",
    "Technologie": "Technology",
    "ElektrischeKwkLeistung": "Capacity",
    "ThermischeNutzleistung": "Capacity_thermal",
    "Inbetriebnahmedatum": "DateIn",
    "DatumEndgueltigeStilllegung": "DateOut",
    }

    CHP_sel.rename(columns=rename_columns, inplace=True)

    # set missing information to match the powerplant data format
    CHP_sel.loc[:, "Set"] = "CHP"
    CHP_sel.loc[:, "Country"] = "DE"
    CHP_sel.loc[:, "Efficiency"] = ""

    # change date format
    CHP_sel.loc[CHP_sel["DateIn"].notnull(), "DateIn"] = CHP_sel.loc[CHP_sel["DateIn"].notnull(), "DateIn"].str[:4].astype(int)
    CHP_sel.loc[CHP_sel["DateOut"].notnull(), "DateOut"] = CHP_sel.loc[CHP_sel["DateOut"].notnull(), "DateOut"].str[:4].astype(int)

    # get location from PLZ
    CHP_sel.loc[:, 'lon'], CHP_sel.loc[:, 'lat'] = zip(*CHP_sel['Postleitzahl'].map(lambda x: geodata.get('DE', {}).get(x, (None, None))))

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

    CHP_sel = CHP_sel.replace({"Fueltype": fueltype})
    CHP_sel = CHP_sel.replace({"Technology": technology})

    def lookup_geodata(missing_plz, geodata):
        for i in range(10):
            plz = missing_plz[:-1] + str(i)
            if plz in geodata['DE']:
                return geodata['DE'][plz]
        for i in range(100):
            if i < 10:
                plz = missing_plz[:-2] + "0" + str(i)
                if plz in geodata['DE']:
                    return geodata['DE'][plz]
            else:
                plz = missing_plz[:-2] + str(i)
                if plz in geodata['DE']:
                    return geodata['DE'][plz]
        return [0, 0, 0]
    
    CHP_sel_empty_lat = CHP_sel[CHP_sel['lat'].isnull()]
    CHP_sel_empty_lat.loc[:, 'lon'] = CHP_sel_empty_lat['Postleitzahl'].apply(lambda plz: lookup_geodata(plz, geodata)[0])
    CHP_sel_empty_lat.loc[:, 'lat'] = CHP_sel_empty_lat['Postleitzahl'].apply(lambda plz: lookup_geodata(plz, geodata)[1])
    CHP_sel.update(CHP_sel_empty_lat)
    
    CHP_sel = CHP_sel.drop(columns=['Postleitzahl'])
    CHP_sel = CHP_sel[['Name', 'Fueltype', 'Technology', 'Set', 'Country', 'Capacity', 'Efficiency', 'DateIn', 'DateOut', 'lat', 'lon', 'Capacity_thermal']]
    return CHP_sel

def calculate_efficiency(CHP_de):
    """
    Calculate the efficiency of the CHP plants depending on Capacity and DateIn.
    Following Triebs et al. (https://doi.org/10.1016/j.ecmx.2020.100068)
    """
    def EXT(cap, year):
        # returns the efficiency for extraction ceondensing turbine
        return ((44/2400) * cap + 0.125 * year - 204.75) / 100

    def BP(cap,year):
        # returns the efficiency for back pressure turbine
        return ((5e-3) * cap + 0.325 * year - 611.75) / 100
    # TODO: differentiate between extraction condensing turbine and back pressure turbine
    CHP_de['Efficiency'] = CHP_de.apply(lambda row: EXT(row['Capacity'], row['DateIn']), axis=1)
        
    return CHP_de
    

if __name__ == "__main__":
    if "snakemake" not in globals():
        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_existing_chp_de", submodule_dir = "./workflow/submodules/pypsa-eur/")
    
    logging.basicConfig(level=snakemake.config["logging"]["level"])

    combustion, biomass = load_dataset(snakemake.input[0])
    
    geodata = load_plz_mapping(snakemake.input[1])

    CHP_de = clean_data(combustion, biomass, geodata)

    CHP_de = calculate_efficiency(CHP_de)
    
    CHP_de.to_csv(snakemake.output.german_chp, index=False)