import pandas as pd
import pypsa
from functools import reduce
from numpy import isclose
import math
import numpy as np
import os

paths = ["workflow/submodules/pypsa-eur/scripts", "../submodules/pypsa-eur/scripts"]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

from prepare_sector_network import prepare_costs

# Defining global varibales

TWh2PJ = 3.6
MWh2TJ = 3.6e-3 
MW2GW = 1e-3
t2Mt = 1e-6

MWh2GJ = 3.6
TWh2PJ = 3.6
MWh2PJ = 3.6e-6


def _get_oil_fossil_fraction(n, region, kwargs):
    total_oil_supply =  n.statistics.supply(
        bus_carrier="oil", **kwargs
    ).drop("Store").groupby("carrier").sum()

    oil_fossil_fraction = (
        total_oil_supply.get("oil")
        / total_oil_supply.sum()
    )

    return oil_fossil_fraction

def __get_oil_fossil_fraction(n, region, kwargs):
    if "DE" in region:
        total_oil_supply =  n.statistics.supply(
            bus_carrier="oil", **kwargs
        ).groupby("name").sum().get([
            "DE oil",
            "DE renewable oil -> DE oil",
            "EU renewable oil -> DE oil",
        ])

    else:
        total_oil_supply =  n.statistics.supply(
            bus_carrier="oil", **kwargs
        ).groupby("name").sum().get([
            "EU oil",
            "DE renewable oil -> EU oil",
            "EU renewable oil -> EU oil",
        ])

    oil_fossil_fraction = (
        total_oil_supply[~total_oil_supply.index.str.contains("renewable")].sum()
        / total_oil_supply.sum()
    )

    return oil_fossil_fraction

def _get_t_sum(df, df_t, carrier, region, snapshot_weightings, port):
    if type(carrier) == list:
        return sum(
            [
                _get_t_sum(
                    df, df_t, car, region, snapshot_weightings, port
                ) for car in carrier
            ]            
        )
    idx = df[df.carrier == carrier].filter(like=region, axis=0).index

    return df_t[port][idx].multiply(
        snapshot_weightings,
        axis=0,
    ).values.sum()

def sum_load(n, carrier, region):
    return MWh2PJ * _get_t_sum(
        n.loads,
        n.loads_t,
        carrier,
        region,
        n.snapshot_weightings.generators,
        "p",
    )   

def sum_co2(n, carrier, region):
    if type(carrier) == list:
        return sum([sum_co2(n, car, region) for car in carrier])
    try:
        port = n.links.groupby(
            "carrier"
        ).first().loc[
            carrier
        ].filter(
            like="bus"
        ).tolist().index("co2 atmosphere")
    except KeyError:
        print(
            "Warning: carrier `", carrier, "` not found in network.links.carrier!",
            sep="")
        return 0

    return -1 * t2Mt * _get_t_sum(
        n.links,
        n.links_t,
        carrier,
        region,
        n.snapshot_weightings.generators,
        f"p{port}",
    )


def get_total_co2(n, region):
    # including international bunker fuels and negative emissions 
    df = n.links.filter(like=region, axis=0)
    co2 = 0
    for port in [col[3:] for col in df if col.startswith("bus")]:
        links = df.index[df[f"bus{port}"] == "co2 atmosphere"]
        co2 -= n.links_t[f"p{port}"][links].multiply(
            n.snapshot_weightings.generators,
            axis=0,
        ).values.sum()
    return t2Mt * co2


def get_capacities(n, region):
    return _get_capacities(n, region, n.statistics.optimal_capacity)

def get_installed_capacities(n, region):
    return _get_capacities(n, region, 
        n.statistics.installed_capacity, cap_string="Installed Capacity|")

def get_capacity_additions_simple(n, region):
    caps = get_capacities(n, region)
    incaps = get_installed_capacities(n, region)
    return pd.Series(data = caps.values - incaps.values,
                     index = "Capacity Additions" + caps.index.str[8:])

def get_capacity_additions(n, region):
    def _f(*args, **kwargs):
        return n.statistics.optimal_capacity(*args, **kwargs).sub(
            n.statistics.installed_capacity(*args, **kwargs), fill_value=0)
    return _get_capacities(n, region, _f, cap_string="Capacity Additions Sub|")

def get_capacity_additions_nstat(n, region):
    def _f(*args, **kwargs):
        kwargs.pop("storage", None)
        return n.statistics.expanded_capacity(*args, **kwargs)
    return _get_capacities(n, region, _f, cap_string="Capacity Additions Nstat|")

def _get_capacities(n, region, cap_func, cap_string="Capacity|"):

    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }


    var = pd.Series()

    capacities_electricity = cap_func(
        bus_carrier=["AC", "low voltage"],
        **kwargs,
    ).filter(like=region).groupby("carrier").sum().drop( 
        # transmission capacities
        ["AC", "DC", "electricity distribution grid"],
        errors="ignore",
    ).multiply(MW2GW)

    var[cap_string + "Electricity|Biomass|w/ CCS"] = \
        capacities_electricity.get('urban central solid biomass CHP CC', 0)
    
    var[cap_string + "Electricity|Biomass|w/o CCS"] = \
        capacities_electricity.get('urban central solid biomass CHP', 0)

    var[cap_string + "Electricity|Biomass|Solids"] = \
        var[[
            cap_string + "Electricity|Biomass|w/ CCS",
            cap_string + "Electricity|Biomass|w/o CCS",
        ]].sum()

    # Ariadne does no checks, so we implement our own?
    assert var[cap_string + "Electricity|Biomass|Solids"] == \
        capacities_electricity.filter(like="solid biomass").sum()

    var[cap_string + "Electricity|Biomass"] = \
        var[cap_string + "Electricity|Biomass|Solids"]


    var[cap_string + "Electricity|Coal|Hard Coal"] = \
        capacities_electricity.get('coal', 0)                                              

    var[cap_string + "Electricity|Coal|Lignite"] = \
        capacities_electricity.get('lignite', 0)
    
    # var[cap_string + "Electricity|Coal|Hard Coal|w/ CCS"] = 
    # var[cap_string + "Electricity|Coal|Hard Coal|w/o CCS"] = 
    # var[cap_string + "Electricity|Coal|Lignite|w/ CCS"] = 
    # var[cap_string + "Electricity|Coal|Lignite|w/o CCS"] = 
    # var[cap_string + "Electricity|Coal|w/ CCS"] = 
    # var[cap_string + "Electricity|Coal|w/o CCS"] = 
    # Q: CCS for coal Implemented, but not activated, should we use it?
    # !: No, because of Kohleausstieg
    # > config: coal_cc


    var[cap_string + "Electricity|Coal"] = \
        var[[
            cap_string + "Electricity|Coal|Lignite",
            cap_string + "Electricity|Coal|Hard Coal",
        ]].sum()

    # var[cap_string + "Electricity|Gas|CC|w/ CCS"] =
    # var[cap_string + "Electricity|Gas|CC|w/o CCS"] =  
    # ! Not implemented, rarely used   

    var[cap_string + "Electricity|Gas|CC"] = \
        capacities_electricity.get('CCGT')
    
    var[cap_string + "Electricity|Gas|OC"] = \
        capacities_electricity.get('OCGT')
    
    var[cap_string + "Electricity|Gas|w/ CCS"] =  \
        capacities_electricity.get('urban central gas CHP CC',0)  
    
    var[cap_string + "Electricity|Gas|w/o CCS"] =  \
        capacities_electricity.get('urban central gas CHP',0) + \
        var[[
            cap_string + "Electricity|Gas|CC",
            cap_string + "Electricity|Gas|OC",
        ]].sum()
    

    var[cap_string + "Electricity|Gas"] = \
        var[[
            cap_string + "Electricity|Gas|w/ CCS",
            cap_string + "Electricity|Gas|w/o CCS",
        ]].sum()

    # var[cap_string + "Electricity|Geothermal"] = 
    # ! Not implemented

    var[cap_string + "Electricity|Hydro"] = \
        pd.Series({
            c: capacities_electricity.get(c) 
            for c in ["ror", "hydro"]
        }).sum()
    # Q!: Not counting PHS here, because it is a true storage,
    # as opposed to hydro
     
    # var[cap_string + "Electricity|Hydrogen|CC"] = 
    # ! Not implemented
    # var[cap_string + "Electricity|Hydrogen|OC"] = 
    # Q: "H2-turbine"
    # Q: What about retrofitted gas power plants? -> Lisa

    var[cap_string + "Electricity|Hydrogen|FC"] = \
        capacities_electricity.get("H2 Fuel Cell")

    var[cap_string + "Electricity|Hydrogen"] = \
        var[cap_string + "Electricity|Hydrogen|FC"]

    # var[cap_string + "Electricity|Non-Renewable Waste"] = 
    # ! Not implemented

    var[cap_string + "Electricity|Nuclear"] = \
        capacities_electricity.get("nuclear", 0)

    # var[cap_string + "Electricity|Ocean"] = 
    # ! Not implemented

    # var[cap_string + "Electricity|Oil|w/ CCS"] = 
    # var[cap_string + "Electricity|Oil|w/o CCS"] = 
    # ! Not implemented

    var[cap_string + "Electricity|Oil"] = \
        capacities_electricity.get("oil")


    var[cap_string + "Electricity|Solar|PV|Rooftop"] = \
        capacities_electricity.get("solar rooftop", 0)
    
    var[cap_string + "Electricity|Solar|PV|Open Field"] = \
        capacities_electricity.get("solar") 

    var[cap_string + "Electricity|Solar|PV"] = \
        var[[
            cap_string + "Electricity|Solar|PV|Open Field",
            cap_string + "Electricity|Solar|PV|Rooftop",
        ]].sum()
    
    # var[cap_string + "Electricity|Solar|CSP"] = 
    # ! not implemented

    var[cap_string + "Electricity|Solar"] = \
        var[cap_string + "Electricity|Solar|PV"]
    
    var[cap_string + "Electricity|Wind|Offshore"] = \
        capacities_electricity.get(
            ["offwind", "offwind-ac", "offwind-dc"]
        ).sum()
    # !: take care of "offwind" -> "offwind-ac"/"offwind-dc"

    var[cap_string + "Electricity|Wind|Onshore"] = \
        capacities_electricity.get("onwind")
    
    var[cap_string + "Electricity|Wind"] = \
        capacities_electricity.filter(like="wind").sum()
    
    assert var[cap_string + "Electricity|Wind"] == \
        var[[
            cap_string + "Electricity|Wind|Offshore",
            cap_string + "Electricity|Wind|Onshore",
        ]].sum()


    # var[cap_string + "Electricity|Storage Converter|CAES"] = 
    # ! Not implemented

    var[cap_string + "Electricity|Storage Converter|Hydro Dam Reservoir"] = \
        capacities_electricity.get('hydro',0)
    
    var[cap_string + "Electricity|Storage Converter|Pump Hydro"] = \
        capacities_electricity.get('PHS',0)

    var[cap_string + "Electricity|Storage Converter|Stationary Batteries"] = \
        capacities_electricity.get("battery discharger",0) + \
        capacities_electricity.get("home battery discharger",0)

    var[cap_string + "Electricity|Storage Converter|Vehicles"] = \
        capacities_electricity.get("V2G", 0)
    
    var[cap_string + "Electricity|Storage Converter"] = \
        var[[
            cap_string + "Electricity|Storage Converter|Hydro Dam Reservoir",
            cap_string + "Electricity|Storage Converter|Pump Hydro",
            cap_string + "Electricity|Storage Converter|Stationary Batteries",
            cap_string + "Electricity|Storage Converter|Vehicles",
        ]].sum()
    

    storage_capacities = cap_func(
        storage=True,
        **kwargs,
    ).filter(like=region).groupby("carrier").sum().multiply(MW2GW)
    # var[cap_string + "Electricity|Storage Reservoir|CAES"] =
    # ! Not implemented
     
    var[cap_string + "Electricity|Storage Reservoir|Hydro Dam Reservoir"] = \
        storage_capacities.get("hydro")

    var[cap_string + "Electricity|Storage Reservoir|Pump Hydro"] = \
        storage_capacities.get("PHS")
    
    var[cap_string + "Electricity|Storage Reservoir|Stationary Batteries"] = \
        pd.Series({
            c: storage_capacities.get(c) 
            for c in ["battery", "home battery"]
        }).sum()
    
    var[cap_string + "Electricity|Storage Reservoir|Vehicles"] = \
        storage_capacities.get("Li ion", 0) 

    var[cap_string + "Electricity|Storage Reservoir"] = \
        var[[
            cap_string + "Electricity|Storage Reservoir|Hydro Dam Reservoir",
            cap_string + "Electricity|Storage Reservoir|Pump Hydro",
            cap_string + "Electricity|Storage Reservoir|Stationary Batteries",
            cap_string + "Electricity|Storage Reservoir|Vehicles",
        ]].sum()


    var[cap_string + "Electricity"] = \
            var[[
            cap_string + "Electricity|Wind",
            cap_string + "Electricity|Solar",
            cap_string + "Electricity|Oil",
            cap_string + "Electricity|Coal",
            cap_string + "Electricity|Gas",
            cap_string + "Electricity|Biomass",
            cap_string + "Electricity|Hydro",
            cap_string + "Electricity|Hydrogen",
            cap_string + "Electricity|Nuclear",
            ]].sum()

    # Test if we forgot something
    _drop_idx = [
        col for col in [
            "PHS",
            "battery discharger",
            "home battery discharger",
            "V2G",
        ] if col in capacities_electricity.index
    ]
    assert isclose(
        var[cap_string + "Electricity"],
        capacities_electricity.drop(_drop_idx).sum(),
    )

    capacities_heat = cap_func(
        bus_carrier=[
            "urban central heat",
            "urban decentral heat",
            "rural heat"
        ],
        **kwargs,
    ).filter(like=region).groupby("carrier").sum().drop(
        ["urban central heat vent"],
        errors="ignore", # drop existing labels or do nothing
    ).multiply(MW2GW)


    var[cap_string + "Heat|Solar thermal"] = \
        capacities_heat.filter(like="solar thermal").sum()
    # TODO Ariadne DB distinguishes between Heat and Decentral Heat!
    # We should probably change all capacities here?!

    # !!! Missing in the Ariadne database
    #  We could be much more detailed for the heat sector (as for electricity)
    # if desired by Ariadne
    #
    var[cap_string + "Heat|Biomass|w/ CCS"] = \
        capacities_heat.get('urban central solid biomass CHP CC',0) 
    var[cap_string + "Heat|Biomass|w/o CCS"] = \
        capacities_heat.get('urban central solid biomass CHP') \
        +  capacities_heat.filter(like="biomass boiler").sum()
    
    var[cap_string + "Heat|Biomass"] = \
        var[cap_string + "Heat|Biomass|w/ CCS"] + \
        var[cap_string + "Heat|Biomass|w/o CCS"]

    assert isclose(
        var[cap_string + "Heat|Biomass"],
        capacities_heat.filter(like="biomass").sum()
    )
    
    var[cap_string + "Heat|Resistive heater"] = \
        capacities_heat.filter(like="resistive heater").sum()
    
    var[cap_string + "Heat|Processes"] = \
        pd.Series({c: capacities_heat.get(c) for c in [
                "Fischer-Tropsch",
                "H2 Electrolysis",
                "H2 Fuel Cell",
                "Sabatier",
                "methanolisation",
        ]}).sum()

    # !!! Missing in the Ariadne database

    var[cap_string + "Heat|Gas"] = \
        capacities_heat.filter(like="gas boiler").sum() \
        + capacities_heat.filter(like="gas CHP").sum()
    
    # var[cap_string + "Heat|Geothermal"] =
    # ! Not implemented 

    var[cap_string + "Heat|Heat pump"] = \
        capacities_heat.filter(like="heat pump").sum()

    var[cap_string + "Heat|Oil"] = \
        capacities_heat.filter(like="oil boiler").sum()

    var[cap_string + "Heat|Storage Converter"] = \
        capacities_heat.filter(like="water tanks discharger").sum()

    storage_capacities = cap_func(
        storage=True,
        **kwargs,
    ).filter(like=region).groupby("carrier").sum().multiply(MW2GW)

    var[cap_string + "Heat|Storage Reservoir"] = \
        storage_capacities.filter(like="water tanks").sum()

    var[cap_string + "Heat"] = (
        var[cap_string + "Heat|Solar thermal"] +
        var[cap_string + "Heat|Resistive heater"] +
        var[cap_string + "Heat|Biomass"] +
        var[cap_string + "Heat|Oil"] +
        var[cap_string + "Heat|Gas"] +
        var[cap_string + "Heat|Processes"] +
        #var[cap_string + "Heat|Hydrogen"] +
        var[cap_string + "Heat|Heat pump"]
    )

    assert isclose(
        var[cap_string + "Heat"],
        capacities_heat[
            # exclude storage converters (i.e., dischargers)
            ~capacities_heat.index.str.contains("discharger")
        ].sum()
    )

    capacities_h2 = cap_func(
        bus_carrier="H2",
        **kwargs,
    ).filter(
        like=region
    ).groupby("carrier").sum().multiply(MW2GW)

    var[cap_string + "Hydrogen|Gas|w/ CCS"] = \
        capacities_h2.get("SMR CC",0)
    
    var[cap_string + "Hydrogen|Gas|w/o CCS"] = \
        capacities_h2.get("SMR",0)
    
    var[cap_string + "Hydrogen|Gas"] = \
        capacities_h2.filter(like="SMR").sum()
    
    assert var[cap_string + "Hydrogen|Gas"] == \
        var[cap_string + "Hydrogen|Gas|w/ CCS"] + \
        var[cap_string + "Hydrogen|Gas|w/o CCS"] 
    
    var[cap_string + "Hydrogen|Electricity"] = \
        capacities_h2.get("H2 Electrolysis", 0)

    var[cap_string + "Hydrogen"] = (
        var[cap_string + "Hydrogen|Electricity"]
        + var[cap_string + "Hydrogen|Gas"]
    )
    assert isclose(
        var[cap_string + "Hydrogen"],
        capacities_h2.reindex([
            "H2 Electrolysis",
            "SMR",
            "SMR CC",
        ]).sum(), # if technology not build, reindex returns NaN
    )

    storage_capacities = cap_func(
        storage=True,
        **kwargs,
    ).filter(like=region).groupby("carrier").sum().multiply(MW2GW)

    var[cap_string + "Hydrogen|Reservoir"] = \
        storage_capacities.get("H2")



    capacities_gas = cap_func(
        bus_carrier="gas",
        **kwargs,
    ).filter(
        like=region
    ).groupby("carrier").sum().drop(
        # Drop Import (Generator, gas), Storage (Store, gas), 
        # and Transmission capacities
        ["gas", "gas pipeline", "gas pipeline new"],
        errors="ignore",
    ).multiply(MW2GW)

    var[cap_string + "Gases|Hydrogen"] = \
        capacities_gas.get("Sabatier", 0)
    
    var[cap_string + "Gases|Biomass"] = \
        capacities_gas.reindex([
            "biogas to gas",
            "biogas to gas CC",
        ]).sum()

    var[cap_string + "Gases"] = (
        var[cap_string + "Gases|Hydrogen"] +
        var[cap_string + "Gases|Biomass"] 
    )

    assert isclose(
        var[cap_string + "Gases"],
        capacities_gas.sum(),
    )


    capacities_liquids = cap_func(
        bus_carrier="renewable oil",
        **kwargs,
    ).filter(
        like=region
    ).groupby("carrier").sum().multiply(MW2GW)
    #
    var[cap_string + "Liquids|Hydrogen"] = \
    var[cap_string + "Liquids"] = \
        capacities_liquids.get("Fischer-Tropsch",0) 
    
    capacities_methanol = cap_func(
        bus_carrier="methanol",
        **kwargs,
    ).filter(
        like=region
    ).groupby("carrier").sum().multiply(MW2GW)
    #
    var[cap_string + "Methanol"] = \
        capacities_methanol.get("methanolisation", 0)
    
    return var 

def get_primary_energy(n, region):
    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }

    var = pd.Series()

    oil_fossil_fraction = _get_oil_fossil_fraction(n, region, kwargs)
    
    oil_usage = n.statistics.withdrawal(
        bus_carrier="oil", 
        **kwargs
    ).filter(
        like=region
    ).groupby(
        "carrier"
    ).sum().multiply(oil_fossil_fraction).multiply(MWh2PJ)

    ## Primary Energy

    var["Primary Energy|Oil|Heat"] = \
        oil_usage.filter(like="oil boiler").sum()

    
    var["Primary Energy|Oil|Electricity"] = \
        oil_usage.get("oil", 0)
    # This will get the oil store as well, but it should be 0
    
    var["Primary Energy|Oil"] = (
        var["Primary Energy|Oil|Electricity"] 
        + var["Primary Energy|Oil|Heat"] 
        + oil_usage.reindex(
            [
                "land transport oil",
                "agriculture machinery oil",
                "shipping oil",
                "kerosene for aviation",
                "naphtha for industry"
            ],
        ).sum()
    )   
    assert isclose(var["Primary Energy|Oil"], oil_usage.sum())

    regional_gas_supply = n.statistics.supply(
        bus_carrier="gas", 
        **kwargs,
    ).filter(
        like=region
    ).groupby(
        ["component", "carrier"]
    ).sum().drop([
        "Store",
        ("Link", "gas pipeline"),
        ("Link", "gas pipeline new"),
    ])

    gas_fossil_fraction = (
        regional_gas_supply.get("Generator").get("gas")
        / regional_gas_supply.sum()
    )
    # Eventhough biogas gets routed through the EU gas bus,
    # it should be counted separately as Primary Energy|Biomass
    gas_usage = n.statistics.withdrawal(
        bus_carrier="gas", 
        **kwargs,
    ).filter(
        like=region
    ).groupby(
        ["component", "carrier"],
    ).sum().drop([
        "Store",
        ("Link", "gas pipeline"),
        ("Link", "gas pipeline new"),
    ]).groupby(
        "carrier"
    ).sum().multiply(gas_fossil_fraction).multiply(MWh2PJ)


    gas_CHP_usage = n.statistics.withdrawal(
        bus_carrier="gas", 
        **kwargs,
    ).filter(
        like=region
    ).filter(
        like="gas CHP"
    ).multiply(gas_fossil_fraction).multiply(MWh2PJ)

    gas_CHP_E_to_H =  (
        n.links.loc[gas_CHP_usage.index.get_level_values("name")].efficiency 
        / n.links.loc[gas_CHP_usage.index.get_level_values("name")].efficiency2
    )

    gas_CHP_E_fraction =  gas_CHP_E_to_H * (1 / (gas_CHP_E_to_H + 1))

    var["Primary Energy|Gas|Heat"] = \
        gas_usage.filter(like="gas boiler").sum() + gas_CHP_usage.multiply(
            1 - gas_CHP_E_fraction
        ).values.sum()
    
    var["Primary Energy|Gas|Electricity"] = \
        gas_usage.reindex(
            [
                'CCGT',
                'OCGT',
            ],
        ).sum() + gas_CHP_usage.multiply(
            gas_CHP_E_fraction
        ).values.sum()

    var["Primary Energy|Gas|Hydrogen"] = \
        gas_usage.filter(like="SMR").sum()
    
    var["Primary Energy|Gas"] = (
        var["Primary Energy|Gas|Heat"]
        + var["Primary Energy|Gas|Electricity"]
        + var["Primary Energy|Gas|Hydrogen"] 
        + gas_usage.filter(like="gas for industry").sum()
    )

    assert isclose(
        var["Primary Energy|Gas"],
        gas_usage.sum(),
    )
    # ! There are CC sub-categories that could be used

    coal_usage = n.statistics.withdrawal(
        bus_carrier=["lignite", "coal"], 
        **kwargs,
    ).filter(
        like=region
    ).groupby(
        "carrier"
    ).sum().multiply(MWh2PJ)

    var["Primary Energy|Coal|Hard Coal"] = \
        coal_usage.get("coal", 0)

    var["Primary Energy|Coal|Lignite"] = \
        coal_usage.get("lignite", 0)
    
    var["Primary Energy|Coal|Electricity"] = \
        var["Primary Energy|Coal|Hard Coal"] + \
        var["Primary Energy|Coal|Lignite"]
    
    var["Primary Energy|Coal"] = (
        var["Primary Energy|Coal|Electricity"] 
        + coal_usage.get("coal for industry", 0)
    )
    
    assert isclose(var["Primary Energy|Coal"], coal_usage.sum())

    var["Primary Energy|Fossil"] = (
        var["Primary Energy|Coal"]
        + var["Primary Energy|Gas"]
        + var["Primary Energy|Oil"]
    )

    biomass_usage = n.statistics.withdrawal(
        bus_carrier=["solid biomass", "biogas"], 
        **kwargs,
    ).filter(
        like=region
    ).groupby(
        "carrier"
    ).sum().multiply(MWh2PJ)

    biomass_CHP_usage = n.statistics.withdrawal(
        bus_carrier="solid biomass", 
        **kwargs,
    ).filter(
        like=region
    ).filter(
        like="CHP"
    ).multiply(MWh2PJ)

    biomass_CHP_E_to_H =  (
        n.links.loc[biomass_CHP_usage.index.get_level_values("name")].efficiency 
        / n.links.loc[biomass_CHP_usage.index.get_level_values("name")].efficiency2
    )

    biomass_CHP_E_fraction =  biomass_CHP_E_to_H * (1 / (biomass_CHP_E_to_H + 1))
    
    var["Primary Energy|Biomass|w/ CCS"] = \
        biomass_usage[biomass_usage.index.str.contains("CC")].sum()
    
    var["Primary Energy|Biomass|w/o CCS"] = \
        biomass_usage[~biomass_usage.index.str.contains("CC")].sum()
    
    var["Primary Energy|Biomass|Electricity"] = \
        biomass_CHP_usage.multiply(
            biomass_CHP_E_fraction
        ).values.sum()
    var["Primary Energy|Biomass|Heat"] = \
        biomass_usage.filter(like="boiler").sum() + biomass_CHP_usage.multiply(
            1 - biomass_CHP_E_fraction
        ).values.sum()
    
    # var["Primary Energy|Biomass|Gases"] = \
    # In this case Gases are only E-Fuels in AriadneDB
    # Not possibly in an easy way because biogas to gas goes to the
    # gas bus, where it mixes with fossil imports
    
    var["Primary Energy|Biomass"] = (
        var["Primary Energy|Biomass|Electricity"]
        + var["Primary Energy|Biomass|Heat"]
        + biomass_usage.filter(like="solid biomass for industry").sum()
        + biomass_usage.filter(like="biogas to gas").sum()
    )
    
        
    # assert isclose(
    #     var["Primary Energy|Biomass"],
    #     biomass_usage.sum(),
    # )

    var["Primary Energy|Nuclear"] = \
        n.statistics.withdrawal(
            bus_carrier=["uranium"], 
            **kwargs,
        ).filter(
            like=region
        ).groupby(
            "carrier"
        ).sum().multiply(MWh2PJ).get("nuclear", 0)


    # ! This should basically be equivalent to secondary energy
    renewable_electricity = n.statistics.supply(
        bus_carrier=["AC", "low voltage"],
        **kwargs,
    ).drop([
        # Assuming renewables are only generators and StorageUnits 
        "Link", "Line"
    ]).filter(like=region).groupby("carrier").sum().multiply(MWh2PJ)

    
    solar_thermal_heat = n.statistics.supply(
        bus_carrier=[
            "urban decentral heat", 
            "urban central heat", 
            "rural heat",
        ],
        **kwargs,
    ).filter(
        like=region
    ).groupby("carrier").sum().filter(
        like="solar thermal"
    ).multiply(MWh2PJ).sum()

    var["Primary Energy|Hydro"] = \
        renewable_electricity.get([
            "ror", "PHS", "hydro",
        ]).sum()
    
    var["Primary Energy|Solar"] = \
        renewable_electricity.filter(like="solar").sum() + \
        solar_thermal_heat

        
    var["Primary Energy|Wind"] = \
        renewable_electricity.filter(like="wind").sum()

    assert isclose(
        renewable_electricity.sum(),
        (
            var["Primary Energy|Hydro"] 
            + var["Primary Energy|Solar"] 
            + var["Primary Energy|Wind"]
        )
    )
    # Primary Energy|Other
    # Not implemented

    return var


def get_secondary_energy(n, region):
    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }
    var = pd.Series()

    electricity_supply = n.statistics.supply(
        bus_carrier=["low voltage", "AC"], **kwargs
    ).filter(like=region).groupby(
        ["carrier"]
    ).sum().multiply(MWh2PJ).drop(
        ["AC", "DC", "electricity distribution grid"],
    )

    var["Secondary Energy|Electricity|Coal|Hard Coal"] = \
        electricity_supply.get("coal", 0)
    
    var["Secondary Energy|Electricity|Coal|Lignite"] = \
        electricity_supply.get("lignite", 0)
    
    var["Secondary Energy|Electricity|Coal"] = (
        var["Secondary Energy|Electricity|Coal|Hard Coal"] 
        + var["Secondary Energy|Electricity|Coal|Lignite"]
    )
    
    var["Secondary Energy|Electricity|Oil"] = \
        electricity_supply.get("oil", 0)
    
    var["Secondary Energy|Electricity|Gas"] = \
        electricity_supply.reindex(
            [
                'CCGT',
                'OCGT',
                'urban central gas CHP',
                'urban central gas CHP CC',
            ],
        ).sum()
    
    var["Secondary Energy|Electricity|Fossil"] = (
        var["Secondary Energy|Electricity|Gas"]
        + var["Secondary Energy|Electricity|Oil"]
        + var["Secondary Energy|Electricity|Coal"]
    )

    var["Secondary Energy|Electricity|Biomass|w/o CCS"] = \
        electricity_supply.get('urban central solid biomass CHP', 0)
    var["Secondary Energy|Electricity|Biomass|w/ CCS"] = \
        electricity_supply.get('urban central solid biomass CHP CC', 0)
    var["Secondary Energy|Electricity|Biomass"] = (
        var["Secondary Energy|Electricity|Biomass|w/o CCS"] 
        + var["Secondary Energy|Electricity|Biomass|w/ CCS"] 
    )
    # ! Biogas to gas should go into this category
    # How to do that? (trace e.g., biogas to gas -> CCGT)
    # If so: Should double counting with |Gas be avoided?
    # -> Might use gas_fossil_fraction just like above  


    var["Secondary Energy|Electricity|Hydro"] = (
        electricity_supply.get("hydro")
        + electricity_supply.get("ror")
    )
    # ! Neglecting PHS here because it is storage infrastructure

    var["Secondary Energy|Electricity|Nuclear"] = \
        electricity_supply.filter(like="nuclear").sum()
    var["Secondary Energy|Electricity|Solar"] = \
        electricity_supply.filter(like="solar").sum()
    var["Secondary Energy|Electricity|Wind|Offshore"] = \
        electricity_supply.get("onwind")
    var["Secondary Energy|Electricity|Wind|Onshore"] = \
        electricity_supply.filter(like="offwind").sum()
    var["Secondary Energy|Electricity|Wind"] = (
        var["Secondary Energy|Electricity|Wind|Offshore"]
        + var["Secondary Energy|Electricity|Wind|Onshore"]
    )
    var["Secondary Energy|Electricity|Non-Biomass Renewables"] = (
        var["Secondary Energy|Electricity|Hydro"]
        + var["Secondary Energy|Electricity|Solar"]
        + var["Secondary Energy|Electricity|Wind"]
    )

    var["Secondary Energy|Electricity|Hydrogen"] = \
        electricity_supply.get("H2 Fuel Cell", 0)
    # ! Add H2 Turbines if they get implemented

    var["Secondary Energy|Electricity|Curtailment"] = \
        n.statistics.curtailment(
            bus_carrier=["AC", "low voltage"], **kwargs
        ).filter(like=region).multiply(MWh2PJ).values.sum()
    

    var["Secondary Energy|Electricity|Storage Losses"] = \
        n.statistics.withdrawal(
            bus_carrier=["AC", "low voltage"], **kwargs
        ).filter(like=region).groupby(["carrier"]).sum().reindex(
            [
                "BEV charger", 
                "battery charger", 
                "home battery charger",
                "PHS",
            ]
        ).subtract(
            n.statistics.supply(
                bus_carrier=["AC", "low voltage"], **kwargs
            ).filter(like=region).groupby(["carrier"]).sum().reindex(
                [
                    "V2G", 
                    "battery discharger", 
                    "home battery discharger",
                    "PHS",
                ]
            )
        ).multiply(MWh2PJ).sum()

    var["Secondary Energy|Electricity|Transmission Losses"] = \
        n.statistics.withdrawal(
            bus_carrier=["AC", "low voltage"], **kwargs
        ).filter(like=region).groupby(["carrier"]).sum().get(
            ["AC", "DC", "electricity distribution grid"]
        ).subtract(
            n.statistics.supply(
                bus_carrier=["AC", "low voltage"], **kwargs
            ).filter(like=region).groupby(["carrier"]).sum().get(
                ["AC", "DC", "electricity distribution grid"]
            )
        ).multiply(MWh2PJ).sum()

    # supply - withdrawal
    # var["Secondary Energy|Electricity|Storage"] = \
    var["Secondary Energy|Electricity"] = (
        var["Secondary Energy|Electricity|Fossil"]
        + var["Secondary Energy|Electricity|Biomass"]
        + var["Secondary Energy|Electricity|Non-Biomass Renewables"]
        + var["Secondary Energy|Electricity|Nuclear"]
        #+ var["Secondary Energy|Electricity|Transmission Losses"]
        #+ var["Secondary Energy|Electricity|Storage Losses"]
        + var["Secondary Energy|Electricity|Hydrogen"]
    )

    assert isclose(
        electricity_supply[
            ~electricity_supply.index.str.contains(
                "PHS"
                "|battery discharger"
                "|home battery discharger"
                "|V2G"
            )
        ].sum(),
        var["Secondary Energy|Electricity"],
    )

    heat_supply = n.statistics.supply(
        bus_carrier=[
            "urban central heat",
            # rural and urban decentral heat do not produce secondary energy
        ], **kwargs
    ).filter(like=region).groupby(
        ["carrier"]
    ).sum().multiply(MWh2PJ)

    var["Secondary Energy|Heat|Gas"] = \
        heat_supply.filter(like="gas").sum()

    var["Secondary Energy|Heat|Biomass"] = \
        heat_supply.filter(like="biomass").sum()
    
    # var["Secondary Energy|Heat|Coal"] = \
    # var["Secondary Energy|Heat|Geothermal"] = \
    # var["Secondary Energy|Heat|Nuclear"] = \
    # var["Secondary Energy|Heat|Other"] = \
    # ! Not implemented

    var["Secondary Energy|Heat|Oil"] = \
        heat_supply.filter(like="oil boiler").sum()
    
    var["Secondary Energy|Heat|Solar"] = \
        heat_supply.filter(like="solar thermal").sum()
    
    var["Secondary Energy|Heat|Electricity|Heat Pumps"] = \
        heat_supply.filter(like="heat pump").sum()
    var["Secondary Energy|Heat|Electricity|Resistive"] = \
        heat_supply.filter(like="resistive heater").sum()
    var["Secondary Energy|Heat|Electricity"] = (
        var["Secondary Energy|Heat|Electricity|Heat Pumps"] 
        + var["Secondary Energy|Heat|Electricity|Resistive"] 
    )
    var["Secondary Energy|Heat|Other"] = \
        heat_supply.reindex(
            [
                "Fischer-Tropsch",
                "H2 Fuel Cell", 
                "H2 Electrolysis",
                "Sabatier",
                "methanolisation",
            ]
        ).sum()
    # TODO remember to specify in comments
    var["Secondary Energy|Heat"] = (
        var["Secondary Energy|Heat|Gas"]
        + var["Secondary Energy|Heat|Biomass"]
        + var["Secondary Energy|Heat|Oil"]
        + var["Secondary Energy|Heat|Solar"]
        + var["Secondary Energy|Heat|Electricity"]
        + var["Secondary Energy|Heat|Other"]
    )
    assert isclose(
        var["Secondary Energy|Heat"],
        heat_supply[
            ~heat_supply.index.str.contains("discharger")
        ].sum()
    )

    hydrogen_production = n.statistics.supply(
        bus_carrier="H2", **kwargs
    ).filter(like=region).groupby(
        ["carrier"]
    ).sum().multiply(MWh2PJ)

    var["Secondary Energy|Hydrogen|Electricity"] = \
        hydrogen_production.get('H2 Electrolysis', 0)

    var["Secondary Energy|Hydrogen|Gas"] = \
        hydrogen_production.get(["SMR","SMR CC"]).sum()

    var["Secondary Energy|Hydrogen"] = (
        var["Secondary Energy|Hydrogen|Electricity"] 
        + var["Secondary Energy|Hydrogen|Gas"]
    )

    assert isclose(
        var["Secondary Energy|Hydrogen"],
        hydrogen_production[
            ~hydrogen_production.index.isin(
                ["H2", "H2 pipeline", "H2 pipeline (Kernnetz)"]
            )
        ].sum()
    )

    oil_fossil_fraction = _get_oil_fossil_fraction(n, region, kwargs)

    
    oil_fuel_usage = n.statistics.withdrawal(
        bus_carrier="oil", 
        **kwargs
    ).filter(
        like=region
    ).groupby(
        "carrier"
    ).sum().multiply(MWh2PJ).reindex(
        [
            "agriculture machinery oil",
            "kerosene for aviation",
            "land transport oil",
            "naphtha for industry",
            "shipping oil"
        ]
    )

    total_oil_fuel_usage = oil_fuel_usage.sum()

    var["Secondary Energy|Liquids|Oil"] = \
        total_oil_fuel_usage * oil_fossil_fraction
    var["Secondary Energy|Liquids|Hydrogen"] = \
        total_oil_fuel_usage * (1 - oil_fossil_fraction)
    
    var["Secondary Energy|Liquids"] = (
        var["Secondary Energy|Liquids|Oil"]
        + var["Secondary Energy|Liquids|Hydrogen"]
    )
    
    methanol_production = n.statistics.supply(
        bus_carrier="methanol", **kwargs
    ).filter(like=region).groupby(
        ["carrier"]
    ).sum().multiply(MWh2PJ)

    assert methanol_production.size <= 1 # only methanolisation

    # var["Production|Chemicals|Methanol"] = \ # here units are Mt/year
    var["Secondary Energy|Other Carrier"] = \
        methanol_production.get("methanolisation", 0)
    # Remeber to specify that Other Carrier == Methanol in Comments Tab


    gas_production = n.statistics.supply(
        bus_carrier="gas", **kwargs
    ).filter(like=region).groupby(
        ["carrier", "component"]
    ).sum().multiply(MWh2PJ).drop(
        ["gas pipeline", "gas pipeline new", ("gas", "Store")]
    ).groupby("carrier").sum() 
    total_gas_production = gas_production.sum()

    gas_fuel_usage = n.statistics.withdrawal(
        bus_carrier="gas", **kwargs
    ).filter(like=region).groupby(
        ["carrier"]
    ).sum().multiply(MWh2PJ).reindex(
        [
            "gas for industry",
            "gas for industry CC",
            "rural gas boiler",
            "urban decentral gas boiler"
        ]
    ) # Building, Transport and Industry sectors

    total_gas_fuel_usage = gas_fuel_usage.sum()

    # Fraction supplied by Hydrogen conversion
    var["Secondary Energy|Gases|Hydrogen"] = (
        total_gas_fuel_usage
        * gas_production.get("Sabatier", 0)
        / total_gas_production
    )
        
    var["Secondary Energy|Gases|Biomass"] = (
        total_gas_fuel_usage
        * gas_production.filter(like="biogas to gas").sum()
        / total_gas_production
    )
        
    var["Secondary Energy|Gases|Natural Gas"] = (
        total_gas_fuel_usage
        * gas_production.get("gas")
        / total_gas_production
    )

    var["Secondary Energy|Gases"] = (
        var["Secondary Energy|Gases|Hydrogen"] 
        + var["Secondary Energy|Gases|Biomass"]
        + var["Secondary Energy|Gases|Natural Gas"]
    )

    assert isclose(
        var["Secondary Energy|Gases"],
        gas_fuel_usage.sum()
    )
        

    return var

def get_final_energy(n, region, _industry_demand, _energy_totals):

    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }

    var = pd.Series()

    energy_totals = _energy_totals.loc[region[0:2]]

    industry_demand = _industry_demand.filter(
        like=region, axis=0,
    ).sum()

    # !: Pypsa-eur does not strictly distinguish between energy and
    # non-energy use

    var["Final Energy|Industry|Electricity"] = \
        industry_demand.get("electricity")
        # or use: sum_load(n, "industry electricity", region)

    var["Final Energy|Industry|Heat"] = \
        industry_demand.get("low-temperature heat")
    
    # var["Final Energy|Industry|Solar"] = \
    # !: Included in |Heat

    # var["Final Energy|Industry|Geothermal"] = \
    # Not implemented

    var["Final Energy|Industry|Gases"] = \
        industry_demand.get("methane")
    # "gas for industry" is now regionally resolved and could be used here

    # var["Final Energy|Industry|Power2Heat"] = \
    # Q: misleading description

    var["Final Energy|Industry|Hydrogen"] = \
        industry_demand.get("hydrogen")
    

    var["Final Energy|Industry|Liquids"] = \
       sum_load(n, "naphtha for industry", region)
    #TODO This is plastics not liquids for industry! Look in industry demand!
    

    # var["Final Energy|Industry|Other"] = \

    var["Final Energy|Industry|Solids"] = \
        industry_demand.get(["coal", "coke", "solid biomass"]).sum()
    
    # Why is AMMONIA zero?
        
    # var["Final Energy|Industry excl Non-Energy Use|Non-Metallic Minerals"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Chemicals"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Steel"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Steel|Primary"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Steel|Secondary"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Pulp and Paper"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Food and Tobacco"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Non-Ferrous Metals"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Engineering"] = \
    # var["Final Energy|Industry excl Non-Energy Use|Vehicle Construction"] = \
    # Q: Most of these could be found somewhere, but are model inputs!

    var["Final Energy|Industry"] = \
        var.get([
            "Final Energy|Industry|Electricity",
            "Final Energy|Industry|Heat",
            "Final Energy|Industry|Gases",
            "Final Energy|Industry|Hydrogen",
            "Final Energy|Industry|Liquids",
            "Final Energy|Industry|Solids",
        ]).sum()

    # Final energy is delivered to the consumers
    low_voltage_electricity = n.statistics.withdrawal(
        bus_carrier="low voltage", 
        **kwargs,
    ).filter(
        like=region,
    ).groupby("carrier").sum().multiply(MWh2PJ)
    
    var["Final Energy|Residential and Commercial|Electricity"] = \
        low_voltage_electricity[
            # carrier does not contain one of the following substrings
            ~low_voltage_electricity.index.str.contains(
                "urban central|industry|agriculture|charger"
                # Excluding chargers (battery and EV)
            )
        ].sum()

    # urban decentral heat and rural heat are delivered as different forms of energy
    # (gas, oil, biomass, ...)
    decentral_heat_withdrawal = n.statistics.withdrawal(
        bus_carrier=["rural heat", "urban decentral heat"], 
        **kwargs,
    ).filter(
        like=region,
    ).groupby("carrier").sum().multiply(MWh2PJ)

    decentral_heat_residential_and_commercial_fraction = (
        decentral_heat_withdrawal.get(
            ["rural heat", "urban decentral heat"]
        ).sum() / decentral_heat_withdrawal.sum()
    )

    decentral_heat_supply_rescom = n.statistics.supply(
        bus_carrier=["rural heat", "urban decentral heat"], 
        **kwargs,
    ).filter(
        like=region,
    ).groupby("carrier").sum().multiply(MWh2PJ).multiply(
        decentral_heat_residential_and_commercial_fraction
    )
    # Dischargers probably should not be considered, to avoid double counting

    var["Final Energy|Residential and Commercial|Heat"] = (
        sum_load(n, "urban central heat", region) # Maybe use n.statistics instead
        + decentral_heat_supply_rescom.filter(like="solar thermal").sum()
    )
        # Assuming for solar thermal secondary energy == Final energy

    var["Final Energy|Residential and Commercial|Gases"] = \
        decentral_heat_supply_rescom.filter(like="gas boiler").sum()

    # var["Final Energy|Residential and Commercial|Hydrogen"] = \
    # ! Not implemented

    var["Final Energy|Residential and Commercial|Liquids"] = \
        decentral_heat_supply_rescom.filter(like="oil boiler").sum()
    
    # var["Final Energy|Residential and Commercial|Other"] = \
    # var["Final Energy|Residential and Commercial|Solids|Coal"] = \
    # ! Not implemented 

    var["Final Energy|Residential and Commercial|Solids"] = \
    var["Final Energy|Residential and Commercial|Solids|Biomass"] = \
        decentral_heat_supply_rescom.filter(like="biomass boiler").sum()

    # Q: Everything else seems to be not implemented

    var["Final Energy|Residential and Commercial"] = (
        var["Final Energy|Residential and Commercial|Electricity"]
        + var["Final Energy|Residential and Commercial|Heat"]
        + var["Final Energy|Residential and Commercial|Gases"]
        + var["Final Energy|Residential and Commercial|Liquids"]
        + var["Final Energy|Residential and Commercial|Solids"]
    )

    # var["Final Energy|Transportation|Other"] = \

    var["Final Energy|Transportation|Electricity"] = \
        sum_load(n, "land transport EV", region)
    
    # var["Final Energy|Transportation|Gases"] = \
    # var["Final Energy|Transportation|Gases|Natural Gas"] = \
    # var["Final Energy|Transportation|Gases|Biomass"] = \
    # var["Final Energy|Transportation|Gases|Efuel"] = \
    # var["Final Energy|Transportation|Gases|Synthetic Fossil"] = \
    # ! Not implemented

    var["Final Energy|Transportation|Hydrogen"] = \
        sum_load(n, "land transport fuel cell", region)
        # ?? H2 for shipping
    

    international_aviation_fraction = \
        energy_totals["total international aviation"] / (
            energy_totals["total domestic aviation"]
            + energy_totals["total international aviation"]
        )
    international_navigation_fraction = \
    energy_totals["total international navigation"] / (
        energy_totals["total domestic navigation"]
        + energy_totals["total international navigation"]
    )

    oil_fossil_fraction = _get_oil_fossil_fraction(n, region, kwargs)

    var["Final Energy|Transportation|Liquids"] = (
        sum_load(n, "land transport oil", region)
        + (
            sum_load(n, "kerosene for aviation", region) 
            * (1 - international_aviation_fraction)
        ) + (
            sum_load(n, ["shipping oil", "shipping methanol"], region)
            * (1 - international_navigation_fraction)
        )
    )
    # var["Final Energy|Transportation|Liquids|Biomass"] = \
    # var["Final Energy|Transportation|Liquids|Synthetic Fossil"] = \
    var["Final Energy|Transportation|Liquids|Petroleum"] = (
        var["Final Energy|Transportation|Liquids"]
        * oil_fossil_fraction
    )
        
    var["Final Energy|Transportation|Liquids|Efuel"] = (
        var["Final Energy|Transportation|Liquids"]
        * (1 - oil_fossil_fraction)
    )


    var["Final Energy|Bunkers|Aviation"] = \
    var["Final Energy|Bunkers|Aviation|Liquids"] = (
        sum_load(n, "kerosene for aviation", region) 
        * international_aviation_fraction
    )


    var["Final Energy|Bunkers|Navigation"] = \
    var["Final Energy|Bunkers|Navigation|Liquids"] = (
        sum_load(n, ["shipping oil", "shipping methanol"], region)
        * international_navigation_fraction
    )

    # var["Final Energy|Bunkers|Navigation|Gases"] = \
    # ! Not implemented
    # var["Final Energy|Bunkers|Navigation|Hydrogen"] = \
    # ! Not used

    var["Final Energy|Bunkers"] = \
        var["Final Energy|Bunkers|Navigation"] \
        + var["Final Energy|Bunkers|Aviation"]

    var["Final Energy|Transportation"] = (
        var["Final Energy|Transportation|Electricity"]
        + var["Final Energy|Transportation|Liquids"]
        + var["Final Energy|Transportation|Hydrogen"]
    )
    
    var["Final Energy|Agriculture|Electricity"] = \
        sum_load(n, "agriculture electricity", region)
    var["Final Energy|Agriculture|Heat"] = \
        sum_load(n, "agriculture heat", region)
    var["Final Energy|Agriculture|Liquids"] = \
        sum_load(n, "agriculture machinery oil", region)
    # var["Final Energy|Agriculture|Gases"] = \
    var["Final Energy|Agriculture"] = (
        var["Final Energy|Agriculture|Electricity"]
        + var["Final Energy|Agriculture|Heat"]
        + var["Final Energy|Agriculture|Liquids"]
    )

    # assert isclose(
    #     var["Final Energy|Agriculture"],
    #     energy_totals.get("total agriculture")
    # ) 
    # It's nice to do these double checks, but it's less
    # straightforward for the other categories
    # !!! TODO this assert is temporarily disbaled because of https://github.com/PyPSA/pypsa-eur/issues/985

    # var["Final Energy"] = \
    # var["Final Energy incl Non-Energy Use incl Bunkers"] = \

    #var["Final Energy|Non-Energy Use|Liquids"] = \
    #var["Final Energy|Non-Energy Use"] = \
    #    industry_demand.get("naphtha") # This is essentially plastics
    # Not sure if this is correct here

    # var["Final Energy|Non-Energy Use|Gases"] = \
    # var["Final Energy|Non-Energy Use|Solids"] = \
    # var["Final Energy|Non-Energy Use|Hydrogen"] = \
    # ! Not implemented 

    var["Final Energy|Electricity"] = (
        var["Final Energy|Agriculture|Electricity"]
        + var["Final Energy|Residential and Commercial|Electricity"]
        + var["Final Energy|Transportation|Electricity"]
        + var["Final Energy|Industry|Electricity"]
    )
    
    # var["Final Energy|Solids"] = \
    # var["Final Energy|Solids|Biomass"] = \
    # var["Final Energy|Gases"] = \
    # var["Final Energy|Liquids"] = \
    var["Final Energy|Heat"] = (
        var["Final Energy|Agriculture|Heat"]
        + var["Final Energy|Residential and Commercial|Heat"]
        + var["Final Energy|Industry|Heat"]
    )
    # var["Final Energy|Solar"] = \
    # var["Final Energy|Hydrogen"] = \

    # var["Final Energy|Geothermal"] = \
    # ! Not implemented

    var["Final Energy incl Non-Energy Use incl Bunkers"] = (
        var["Final Energy|Industry"]
        + var["Final Energy|Residential and Commercial"]
        + var["Final Energy|Agriculture"]
        + var["Final Energy|Transportation"]
        + var["Final Energy|Bunkers"]
    )
        

    # The general problem with final energy is that for most of these categories
    # feedstocks shouls be excluded (i.e., non-energy use)
    # However this is hard to do in PyPSA.
    # TODO nevertheless it would be nice to do exactly that

    return var


def get_emissions(n, region, _energy_totals):
    
    energy_totals = _energy_totals.loc[region[0:2]]

    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }

    var = pd.Series()

    co2_emissions = n.statistics.supply(
        bus_carrier="co2",**kwargs
    ).filter(like=region).groupby("carrier").sum().multiply(t2Mt)  
    

    CHP_emissions = n.statistics.supply(
        bus_carrier="co2",**kwargs
    ).filter(like=region).filter(like="CHP").multiply(t2Mt)

    CHP_E_to_H =  (
        n.links.loc[CHP_emissions.index.get_level_values("name")].efficiency 
        / n.links.loc[CHP_emissions.index.get_level_values("name")].efficiency2
    )

    CHP_E_fraction =  CHP_E_to_H * (1 / (CHP_E_to_H + 1))

    negative_CHP_emissions = n.statistics.withdrawal(
        bus_carrier="co2",**kwargs
    ).filter(like=region).filter(like="CHP").multiply(t2Mt)

    negative_CHP_E_to_H =  (
        n.links.loc[
            negative_CHP_emissions.index.get_level_values("name")
        ].efficiency 
        / n.links.loc[
            negative_CHP_emissions.index.get_level_values("name")
        ].efficiency2
    )

    negative_CHP_E_fraction =  negative_CHP_E_to_H * (
        1 / (negative_CHP_E_to_H + 1)
    )

    co2_negative_emissions = n.statistics.withdrawal(
        bus_carrier="co2",**kwargs
    ).filter(like=region).groupby("carrier").sum().multiply(t2Mt)

    co2_storage = n.statistics.supply(
        bus_carrier ="co2 stored",**kwargs
    ).filter(like=region).groupby("carrier").sum().multiply(t2Mt)    

    var["Carbon Sequestration"] = \
        n.statistics.supply(
            bus_carrier="co2 sequestered",**kwargs
        ).filter(like=region).groupby("carrier").sum().multiply(t2Mt).sum()     

    var["Carbon Sequestration|DACCS"] = \
        var["Carbon Sequestration"] * (
            co2_storage.filter(like="DAC").sum()
            / co2_storage.sum()
        )
    
    var["Carbon Sequestration|BECCS"] = \
        var["Carbon Sequestration"] * (
            co2_storage.filter(like="bio").sum()
            / co2_storage.sum()
        )
    
    var["Carbon Sequestration|Other"] = (
        var["Carbon Sequestration"] 
        - var["Carbon Sequestration|DACCS"]
        - var["Carbon Sequestration|BECCS"]
    )


    var["Emissions|CO2"] = \
        co2_emissions.sum() - co2_negative_emissions.sum()

    # ! LULUCF should also be subtracted (or added??), we get from REMIND, 
    # TODO how to consider it here?
    
    # Make sure these values are about right
    var["Emissions|CO2|Industrial Processes"] = \
        co2_emissions.reindex([
            "process emissions",
            "process emissions CC",
        ]).sum()
    
    # !!! We do not strictly separate fuel combustion emissions from
    # process emissions in industry, so some should go to:
    var["Emissions|CO2|Energy|Demand|Industry"] = \
        co2_emissions.reindex([
            "gas for industry",
            "gas for industry CC",
            "coal for industry"
        ]).sum() - co2_negative_emissions.get(
            "solid biomass for industry CC", 
            0,
        ) 
       
    var["Emissions|CO2|Energy|Demand|Residential and Commercial"] = (
        co2_emissions.filter(like="urban decentral").sum() 
        + co2_emissions.filter(like="rural" ).sum()
    )

    international_aviation_fraction = \
        energy_totals["total international aviation"] / (
            energy_totals["total domestic aviation"]
            + energy_totals["total international aviation"]
        )
    international_navigation_fraction = \
    energy_totals["total international navigation"] / (
        energy_totals["total domestic navigation"]
        + energy_totals["total international navigation"]
    )

    var["Emissions|CO2|Energy|Demand|Transportation"] = (
        co2_emissions.get("land transport oil", 0) + (
            co2_emissions.get("kerosene for aviation") 
            * (1 - international_aviation_fraction)
        ) + (
            co2_emissions.filter(like="shipping").sum()
            * (1 - international_navigation_fraction)
        )
    )
  
    var["Emissions|CO2|Energy|Demand|Bunkers|Aviation"] = (
        co2_emissions.get("kerosene for aviation") 
        * international_aviation_fraction
    )

    var["Emissions|CO2|Energy|Demand|Bunkers|Navigation"] = (
        co2_emissions.filter(like="shipping").sum()
        * international_navigation_fraction
    )
    
    var["Emissions|CO2|Energy|Demand|Bunkers"] = \
        var["Emissions|CO2|Energy|Demand|Bunkers|Aviation"] + \
        var["Emissions|CO2|Energy|Demand|Bunkers|Navigation"]
    
    var["Emissions|CO2|Energy|Demand|Other Sector"] = \
        co2_emissions.get("agriculture machinery oil")
    
    var["Emissions|CO2|Energy|Demand"] = \
        var.get([
            "Emissions|CO2|Energy|Demand|Industry",
            "Emissions|CO2|Energy|Demand|Transportation",
            "Emissions|CO2|Energy|Demand|Residential and Commercial",
            "Emissions|CO2|Energy|Demand|Other Sector"
        ]).sum()
    var["Emissions|CO2|Energy incl Bunkers|Demand"] = \
        var["Emissions|CO2|Energy|Demand"] + \
        var["Emissions|CO2|Energy|Demand|Bunkers"]
    
    var["Emissions|Gross Fossil CO2|Energy|Supply|Electricity"] = \
        co2_emissions.reindex(
            [
                "OCGT",
                "CCGT",
                "coal",
                "lignite",
                "oil",
            ], 
        ).sum() + CHP_emissions.multiply(
            CHP_E_fraction
        ).values.sum()



    var["Emissions|CO2|Energy|Supply|Electricity"] = (
        var["Emissions|Gross Fossil CO2|Energy|Supply|Electricity"]
        - negative_CHP_emissions.multiply(
            negative_CHP_E_fraction
        ).values.sum()
    )

    var["Emissions|Gross Fossil CO2|Energy|Supply|Heat"] = \
        co2_emissions.filter(
            like="urban central"
        ).filter(
            like="boiler" # in 2020 there might be central oil boilers?!
        ).sum() + CHP_emissions.multiply(
            1 - CHP_E_fraction
        ).values.sum()
    

    var["Emissions|CO2|Energy|Supply|Heat"] = (
        var["Emissions|Gross Fossil CO2|Energy|Supply|Heat"]
        - negative_CHP_emissions.multiply(
            1 - negative_CHP_E_fraction
        ).values.sum()
    )

    var["Emissions|CO2|Energy|Supply|Electricity and Heat"] = \
        var["Emissions|CO2|Energy|Supply|Heat"] + \
        var["Emissions|CO2|Energy|Supply|Electricity"]


    var["Emissions|CO2|Energy|Supply|Hydrogen"] = \
        co2_emissions.filter(like="SMR").sum()
    
    var["Emissions|CO2|Energy|Supply|Gases"] = \
        (-1) * co2_negative_emissions.filter(
            like="biogas to gas"
        ).sum()
    
    var["Emissions|CO2|Supply|Non-Renewable Waste"] = \
        co2_emissions.get("naphtha for industry")

    # var["Emissions|CO2|Energy|Supply|Liquids"] = \
    # Our only Liquid production is Fischer-Tropsch
    # -> no emissions in this category

    # var["Emissions|CO2|Energy|Supply|Liquids and Gases"] = \
        # var["Emissions|CO2|Energy|Supply|Liquids"]
        # var["Emissions|CO2|Energy|Supply|Gases"] + \
    
    var["Emissions|CO2|Energy|Supply"] = \
        var["Emissions|CO2|Energy|Supply|Gases"] + \
        var["Emissions|CO2|Energy|Supply|Hydrogen"] + \
        var["Emissions|CO2|Energy|Supply|Electricity and Heat"]
    
    # var["Emissions|CO2|Energy|Supply|Other Sector"] = \   
    # var["Emissions|CO2|Energy|Supply|Solids"] = \ 

    var["Emissions|CO2|Energy"] = \
        var["Emissions|CO2|Energy|Demand"] + \
        var["Emissions|CO2|Energy|Supply"]  
         
    var["Emissions|CO2|Energy incl Bunkers"] = \
        var["Emissions|CO2|Energy incl Bunkers|Demand"] + \
        var["Emissions|CO2|Energy|Supply"]  

    var["Emissions|CO2|Energy and Industrial Processes"] = \
        var["Emissions|CO2|Energy"] + \
        var["Emissions|CO2|Industrial Processes"]

    assert isclose(
        var["Emissions|CO2"],
        (
            var["Emissions|CO2|Energy and Industrial Processes"] 
            + var["Emissions|CO2|Energy|Demand|Bunkers"]
            + var["Emissions|CO2|Supply|Non-Renewable Waste"]
            - co2_negative_emissions.get("DAC", 0)
        )
    )
    return var 

# functions for prices
def get_nodal_flows(n, bus_carrier, region, query='index == index or index != index'):
    """
    Get the nodal flows for a given bus carrier and region.

    Parameters:
        n (pypsa.Network): The PyPSA network object.
        bus_carrier (str): The bus carrier for which to retrieve the nodal flows.
        region (str): The region for which to retrieve the nodal flows.
        query (str, optional): A query string to filter the nodal flows. Defaults to 'index == index or index != index'.

    Returns:
        pandas.DataFrame: The nodal flows for the specified bus carrier and region.
    """

    groupby = n.statistics.groupers.get_name_bus_and_carrier

    result = n.statistics.withdrawal(
        bus_carrier=bus_carrier, 
        groupby=groupby,
        aggregate_time=False,
    ).filter(
        like=region,
        axis=0,
    ).query(query).groupby("bus").sum().T 
    
    return result


def price_load(n, load_carrier, region):
    """
    Calculate the average price of a specific load carrier in a given region.

    Parameters:
    - n (pandas.DataFrame): The network model.
    - load_carrier (str): The load carrier to calculate the price for.
    - region (str): The region to calculate the price in.

    Returns:
    - tuple: A tuple containing the average price and the total load of the specified load carrier in the region.
    """

    load = n.loads[(n.loads.carrier == load_carrier) & (n.loads.bus.str.contains(region))]
    if n.loads_t.p[load.index].values.sum() < 1:
        return np.nan, 0
    result = (n.loads_t.p[load.index] * n.buses_t.marginal_price[load.bus].values).values.sum()
    result /= n.loads_t.p[load.index].values.sum()
    return result, n.loads_t.p[load.index].values.sum()


def costs_gen_generators(n, region, carrier):
    """
    Calculate the cost per unit of generated energy of a generators in a given region.

    Parameters:
    - n (pandas.DataFrame): The network model.
    - region (str): The region to consider.
    - carrier (str): The carrier of the generators.

    Returns:
    - tuple: A tuple containing cost and total generation of the generators.
    """
    
    gens = n.generators[(n.generators.carrier == carrier) 
                        & (n.generators.bus.str.contains(region))]
    gen = n.generators_t.p[gens.index].multiply(
        n.snapshot_weightings.generators, axis="index").sum()
    if gen.empty or gen.sum() < 1:
        return np.nan, 0

    # CAPEX
    capex = (gens.p_nom_opt * gens.capital_cost).sum()

    # OPEX
    opex = (gen * gens.marginal_cost).sum()
              
    result = (capex + opex) / gen.sum()
    return result, gen.sum()


def costs_gen_links(n, region, carrier, gen_bus="p1"):
    """
    Calculate the cost per unit of generated energy from a specific link.

    Parameters:
        n (pypsa.Network): The PyPSA network object.
        region (str): The region to consider for the links.
        carrier (str): The carrier of the links.
        gen_bus (str, optional): The bus where the main generation of the link takes place. Defaults to "p1".

    Returns:
        tuple: A tuple containing the costs per unit of generetad energy and the total generation of the specified generator bus.
    """

    links = n.links[(n.links.carrier == carrier) 
                    & (n.links.index.str.contains(region))]
    gen = abs(n.links_t[gen_bus][links.index].multiply(
        n.snapshot_weightings.generators, axis="index")).sum()
    if gen.empty or gen.sum() < 1:
        return np.nan, 0
    
    # CAPEX
    capex = (links.p_nom_opt * links.capital_cost).sum()

    # OPEX
    input = abs(n.links_t["p0"][links.index].multiply(
        n.snapshot_weightings.generators, axis="index")).sum()
    opex = (input * links.marginal_cost).sum()

    # input costs and output revenues other than main generation @ gen_bus
    sum = 0
    for i in range(0,5):
        if f"p{i}" == gen_bus:
            continue
        elif links.empty:
            break
        elif n.links.loc[links.index][f"bus{i}"].iloc[0] == "":
            break
        else:
            update_cost = (
                    n.links_t[f"p{i}"][links.index] 
                    * n.buses_t.marginal_price[links[f"bus{i}"]].values
                ).multiply(
                    n.snapshot_weightings.generators, axis="index"
                ).values.sum()
            sum = sum + update_cost
              
    result = (capex + opex + sum) / gen.sum()
    return result, gen.sum()


def get_weighted_costs_links(carriers, n, region):
    numerator = 0
    denominator = 0
    
    for c in carriers:   
        cost_gen = costs_gen_links(n, region, c)
        if not math.isnan(cost_gen[0]):
            numerator += cost_gen[0] * cost_gen[1]
            denominator += cost_gen[1]
        
    if denominator == 0:
        return np.nan
    result = numerator / denominator 
    return result

def get_weighted_costs(costs, flows):

    cleaned_costs = []
    cleaned_flows = []

    for cost, flow in zip(costs, flows):
        if not math.isnan(cost) and not math.isnan(flow) and flow != 0:
            cleaned_costs.append(cost)
            cleaned_flows.append(flow)
    
    if not cleaned_costs or not cleaned_flows:
        return np.nan
    
    df_cleaned = pd.DataFrame({'costs': cleaned_costs, 'flows': cleaned_flows})
    result = (df_cleaned["costs"] * df_cleaned["flows"]).sum() / df_cleaned["flows"].sum()
    return result


def get_prices(n, region):
    """
    Calculate the prices of various energy sources in the Ariadne model.

    Parameters:
    - n (PyPSa network): The Ariadne model scenario output.
    - region (str): The region for which the prices are calculated.

    Returns:
    - var (pandas.Series): A series containing the calculated prices.

    This function calculates the prices of different energy sources in the Ariadne model
    based on the nodal flows and marginal prices of the model. The calculated prices are
    stored in a pandas Series object and returned.
    """

    var = pd.Series()

    nodal_flows_lv = get_nodal_flows(
        n, "low voltage", region,
        query = "not carrier.str.contains('agriculture')"
                "& not carrier.str.contains('industry')"
                "& not carrier.str.contains('urban central')"
            )

    nodal_prices_lv = n.buses_t.marginal_price[nodal_flows_lv.columns] 

    # electricity price at the final level in the residential sector. Prices should include the effect of carbon prices.
    var["Price|Final Energy|Residential|Electricity"] = \
        nodal_flows_lv.mul(nodal_prices_lv).values.sum() / nodal_flows_lv.values.sum() / MWh2GJ
    
    # vars: Tier 1, Category: energy(price)

    nodal_flows_bm = get_nodal_flows(n, "solid biomass", region)
    nodal_prices_bm = n.buses_t.marginal_price[nodal_flows_bm.columns]

    # primary energy consumption of purpose-grown bioenergy crops, crop and forestry residue bioenergy, municipal solid waste bioenergy, traditional biomass, including renewable waste
    var["Price|Primary Energy|Biomass"] = \
        nodal_flows_bm.mul(nodal_prices_bm).values.sum() / nodal_flows_bm.values.sum() / MWh2GJ
    
    # Price|Primary Energy|Coal
    # is coal also lignite? -> yes according to michas code (coal for industry is already included as it withdraws from coal bus)
    nf_coal = get_nodal_flows(n, "coal", region)
    nodal_prices_coal = n.buses_t.marginal_price[nf_coal.columns]
    coal_price = nf_coal.mul(nodal_prices_coal).values.sum() / nf_coal.values.sum() if nf_coal.values.sum() > 0 else np.nan

    nf_lignite = get_nodal_flows(n, "lignite", region)
    nodal_prices_lignite = n.buses_t.marginal_price[nf_lignite.columns]
    lignite_price = nf_lignite.mul(nodal_prices_lignite).values.sum() / nf_lignite.values.sum() if nf_lignite.values.sum() > 0 else np.nan

    var["Price|Primary Energy|Coal"] = \
        get_weighted_costs([coal_price, lignite_price], [nf_coal.values.sum(), nf_lignite.values.sum()])/ MWh2GJ
    
    # Price|Primary Energy|Gas
    nodal_flows_gas = get_nodal_flows(n, "gas", region)
    nodal_prices_gas = n.buses_t.marginal_price[nodal_flows_gas.columns]

    var["Price|Primary Energy|Gas"] = \
        nodal_flows_gas.mul(nodal_prices_gas).values.sum()  / nodal_flows_gas.values.sum() / MWh2GJ
    
    # Price|Primary Energy|Oil
    nodal_flows_oil = get_nodal_flows(n, "oil", region)
    nodal_prices_oil = n.buses_t.marginal_price[nodal_flows_oil.columns]

    var["Price|Primary Energy|Oil"] = \
        nodal_flows_oil.mul(nodal_prices_oil).values.sum() / nodal_flows_oil.values.sum() /MWh2GJ

    # Price|Secondary Energy|Electricity
    # electricity price at the secondary level, i.e. for large scale consumers (e.g. aluminum production). Prices should include the effect of carbon prices.

    nodal_flows_ac = get_nodal_flows(
        n, "AC", region,
        query = "not carrier.str.contains('gas')"
            )
    nodal_prices_ac = n.buses_t.marginal_price[nodal_flows_ac.columns]

    var["Price|Secondary Energy|Electricity"] = \
    nodal_flows_ac.mul(nodal_prices_ac).values.sum() / nodal_flows_ac.values.sum() /MWh2GJ

    var["Price|Secondary Energy|Gases|Natural Gas"] = \
        costs_gen_generators(n, region ,"gas")[0] / MWh2GJ

    var["Price|Secondary Energy|Gases|Hydrogen"] = \
        costs_gen_links(n, region, "Sabatier")[0] / MWh2GJ

    var["Price|Secondary Energy|Gases|Biomass"] = \
        costs_gen_links(n, region, "biogas to gas")[0] / MWh2GJ
    
    # Price|Secondary Energy|Gases|Efuel
    # Price for gaseous Efuels at the secondary level, i.e. for large scale consumers. Prices should include the effect of carbon prices.
    # what are gaseous Efuels?
    
    # Price|Secondary Energy|Hydrogen
    nodal_flows_h2 = get_nodal_flows(
        n, "H2", region
        )
    nodal_prices_h2 = n.buses_t.marginal_price[nodal_flows_h2.columns]

    var["Price|Secondary Energy|Hydrogen"] = \
        nodal_flows_h2.mul(nodal_prices_h2).values.sum() / nodal_flows_h2.values.sum() /MWh2GJ  

    # From PIK plots
    # "Price|Final Energy|Residential|Hydrogen" = final energy consumption by the residential sector of hydrogen
    # do we have residential applications for hydrogen?

    nf_gas_residential = get_nodal_flows(
        n, "gas", region,
        query = "carrier.str.contains('rural')"
                "or carrier.str.contains('urban decentral')"
        )
    nodal_prices_gas = n.buses_t.marginal_price[nf_gas_residential.columns]

    # !!! mv much higher: check carbon effect!
    var["Price|Final Energy|Residential|Gases"] = \
        nf_gas_residential.mul(nodal_prices_gas).values.sum() / nf_gas_residential.values.sum() / MWh2GJ  if nf_gas_residential.values.sum() > 0 else np.nan

    # "Price|Final Energy|Residential|Gases|Natural Gas" ?
    # "Price|Final Energy|Residential|Liquids|Biomass" x
    
    var["Price|Final Energy|Residential|Liquids|Oil"] = \
        get_weighted_costs_links(
            ['rural oil boiler', 'urban decentral oil boiler'], 
            n, region) / MWh2GJ

    var["Price|Final Energy|Residential|Liquids"] = \
        var["Price|Final Energy|Residential|Liquids|Oil"]

    var["Price|Final Energy|Residential|Solids|Biomass"] = \
        get_weighted_costs_links(
            ['rural biomass boiler', 'urban decentral biomass boiler'],
            n, region) / MWh2GJ
    
    var["Price|Final Energy|Residential|Solids"] = \
        var["Price|Final Energy|Residential|Solids|Biomass"]

    # "Price|Final Energy|Industry|Electricity"✓

    var["Price|Final Energy|Industry|Gases"] = \
        get_weighted_costs_links(
            ['gas for industry','gas for industry CC'],
            n, region) / MWh2GJ

    # "Price|Final Energy|Industry|Heat"✓

    var["Price|Final Energy|Industry|Liquids"] = \
        price_load(n, "naphtha for industry", region)[0] / MWh2GJ
    
    # "Price|Final Energy|Industry|Hydrogen"✓

    var["Price|Final Energy|Industry|Solids"] = \
        get_weighted_costs_links(
            [ 'solid biomass for industry', 'solid biomass for industry CC', 'coal for industry'],
            n, region) / MWh2GJ

    # Rest Tier 2
    # x
    # Price|Final Energy|Transportation|Liquids|Petroleum
    # Price|Final Energy|Transportation|Liquids|Petroleum|Sales Margin
    # Price|Final Energy|Transportation|Liquids|Petroleum|Transport and Distribution
    # Price|Final Energy|Transportation|Liquids|Petroleum|Carbon Price Component
    # Price|Final Energy|Transportation|Liquids|Petroleum|Other Taxes
    # 'land transport oil' ?

    # x
    # Price|Final Energy|Transportation|Liquids|Diesel
    # Price|Final Energy|Transportation|Liquids|Diesel|Sales Margin
    # Price|Final Energy|Transportation|Liquids|Diesel|Transport and Distribution
    # Price|Final Energy|Transportation|Liquids|Diesel|Carbon Price Component
    # Price|Final Energy|Transportation|Liquids|Diesel|Other Taxes

    # Price|Final Energy|Transportation|Gases|Natural Gas
    # Price|Final Energy|Transportation|Gases|Natural Gas|Sales Margin
    # Price|Final Energy|Transportation|Gases|Natural Gas|Transport and Distribution
    # Price|Final Energy|Transportation|Gases|Natural Gas|Carbon Price Component
    # Price|Final Energy|Transportation|Gases|Natural Gas|Other Taxes

    # x
    # Price|Final Energy|Transportation|Liquids|Biomass
    # Price|Final Energy|Transportation|Liquids|Biomass|Sales Margin
    # Price|Final Energy|Transportation|Liquids|Biomass|Transport and Distribution
    # Price|Final Energy|Transportation|Liquids|Biomass|Other Taxes

    # Price|Final Energy|Transportation|Liquids|Efuel

    df = pd.DataFrame({c: price_load(n, c, region) for c in \
                       ["kerosene for aviation", "shipping methanol", "shipping oil"]})
    
    var["Price|Final Energy|Transportation|Liquids|Efuel"]  = \
        (df.iloc[0]*df.iloc[1]).sum() / df.iloc[1].sum() / MWh2GJ

    # Price|Final Energy|Transportation|Liquids|Efuel|Sales Margin
    # Price|Final Energy|Transportation|Liquids|Efuel|Transport and Distribution
    # Price|Final Energy|Transportation|Liquids|Efuel|Other Taxes

    # Price|Final Energy|Transportation|Gases|Efuel
    # Price|Final Energy|Transportation|Gases|Efuel|Sales Margin
    # Price|Final Energy|Transportation|Gases|Efuel|Transport and Distribution
    # Price|Final Energy|Transportation|Gases|Efuel|Other Taxes

    # Price|Final Energy|Transportation|Hydrogen
    # Price|Final Energy|Transportation|Hydrogen|Sales Margin
    # Price|Final Energy|Transportation|Hydrogen|Transport and Distribution
    # Price|Final Energy|Transportation|Hydrogen|Other Taxes

    # Price|Final Energy|Transportation|Electricity

    var["Price|Final Energy|Transportation|Electricity"] = \
        price_load(n, "land transport EV", region)[0] / (MWh2GJ)
    
    # Price|Final Energy|Transportation|Electricity|Sales Margin
    # Price|Final Energy|Transportation|Electricity|Transport and Distribution
    # Price|Final Energy|Transportation|Electricity|Other Taxes

    # Price|Final Energy|Residential and Commercial|Liquids|Oil

    var["Price|Final Energy|Residential and Commercial|Liquids|Oil"] = \
        get_weighted_costs_links(
            ['rural oil boiler', 'urban decentral oil boiler'],
            n, region) / MWh2GJ

    # Price|Final Energy|Residential and Commercial|Liquids|Oil|Sales Margin
    # Price|Final Energy|Residential and Commercial|Liquids|Oil|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Liquids|Oil|Carbon Price Component
    # Price|Final Energy|Residential and Commercial|Liquids|Oil|Other Taxes

    # Price|Final Energy|Residential and Commercial|Gases|Natural Gas
    # cannot really be reasonably divided from non Natural Gas resources (at least no low hanging fruit :))
    # Price|Final Energy|Residential and Commercial|Gases|Natural Gas|Sales Margin
    # Price|Final Energy|Residential and Commercial|Gases|Natural Gas|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Gases|Natural Gas|Carbon Price Component
    # Price|Final Energy|Residential and Commercial|Gases|Natural Gas|Other Taxes

    # Price|Final Energy|Residential and Commercial|Heat
    nf_rc_heat = get_nodal_flows(
        n, ['urban central heat', 'rural heat', 'urban decentral heat'], region,
        query = "not carrier.str.contains('agriculture')"
                "& not carrier.str.contains('industry')"
                "& not carrier.str.contains('DAC')"
            )

    np_rc_heat = n.buses_t.marginal_price[nf_rc_heat.columns]
    var["Price|Final Energy|Residential and Commercial|Heat"] = \
        nf_rc_heat.mul(np_rc_heat).values.sum() / nf_rc_heat.values.sum() / MWh2GJ

    # Price|Final Energy|Residential and Commercial|Heat|Sales Margin
    # Price|Final Energy|Residential and Commercial|Heat|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Heat|Other Taxes

    # Price|Final Energy|Residential and Commercial|Liquids|Biomass   
    # Price|Final Energy|Residential and Commercial|Liquids|Biomass|Sales Margin
    # Price|Final Energy|Residential and Commercial|Liquids|Biomass|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Liquids|Biomass|Other Taxes

    # Price|Final Energy|Residential and Commercial|Solids|Biomass

    var["Price|Final Energy|Residential and Commercial|Solids|Biomass"] = \
        get_weighted_costs_links(
            ['rural biomass boiler', 'urban decentral biomass boiler'],
            n, region) / MWh2GJ
    
    # Price|Final Energy|Residential and Commercial|Solids|Biomass|Sales Margin
    # Price|Final Energy|Residential and Commercial|Solids|Biomass|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Solids|Biomass|Other Taxes

    # Price|Final Energy|Residential and Commercial|Gases|Biomass x
    # Price|Final Energy|Residential and Commercial|Gases|Biomass|Sales Margin
    # Price|Final Energy|Residential and Commercial|Gases|Biomass|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Gases|Biomass|Other Taxes

    # Price|Final Energy|Residential and Commercial|Liquids|Efuel x
    # Price|Final Energy|Residential and Commercial|Liquids|Efuel|Sales Margin
    # Price|Final Energy|Residential and Commercial|Liquids|Efuel|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Liquids|Efuel|Other Taxes

    # Price|Final Energy|Residential and Commercial|Gases|Efuel x
    # Price|Final Energy|Residential and Commercial|Gases|Efuel|Sales Margin
    # Price|Final Energy|Residential and Commercial|Gases|Efuel|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Gases|Efuel|Other Taxes

    # Price|Final Energy|Residential and Commercial|Hydrogen x
    # Price|Final Energy|Residential and Commercial|Hydrogen|Sales Margin
    # Price|Final Energy|Residential and Commercial|Hydrogen|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Hydrogen|Other Taxes

    var["Price|Final Energy|Residential and Commercial|Electricity"] = \
        var["Price|Final Energy|Residential|Electricity"]

    # Price|Final Energy|Residential and Commercial|Electricity|Sales Margin x
    # Price|Final Energy|Residential and Commercial|Electricity|Transport and Distribution
    # Price|Final Energy|Residential and Commercial|Electricity|Other Taxes
    var["Price|Final Energy|Industry|Electricity"] = \
        price_load(n, "industry electricity", region)[0] / (MWh2GJ)
    
    var["Price|Final Energy|Industry|Heat"] = \
        price_load(n, "low-temperature heat for industry", region)[0] / (MWh2GJ)
        
    var["Price|Final Energy|Industry|Hydrogen"] = \
        price_load(n, "H2 for industry", region)[0] / (MWh2GJ)
    
    var["Price|Final Energy|Industry|Solids|Coal"] = \
        price_load(n, "coal for industry", region)[0] / (MWh2GJ)
    
    # Price|Final Energy|Industry|Solids|Coal|Sales Margin x
    # Price|Final Energy|Industry|Solids|Coal|Transport and Distribution
    # Price|Final Energy|Industry|Solids|Coal|Carbon Price Component
    # Price|Final Energy|Industry|Solids|Coal|Other Taxes

    # var["Price|Final Energy|Industry|Gases|Natural Gas"] ?

    # Price|Final Energy|Industry|Gases|Natural Gas|Sales Margin x
    # Price|Final Energy|Industry|Gases|Natural Gas|Transport and Distribution
    # Price|Final Energy|Industry|Gases|Natural Gas|Carbon Price Component
    # Price|Final Energy|Industry|Gases|Natural Gas|Other Taxes

    # Price|Final Energy|Industry|Heat|Sales Margin x
    # Price|Final Energy|Industry|Heat|Transport and Distribution
    # Price|Final Energy|Industry|Heat|Other Taxes

    # Price|Final Energy|Industry|Liquids|Biomass x
    # Price|Final Energy|Industry|Liquids|Biomass|Sales Margin
    # Price|Final Energy|Industry|Liquids|Biomass|Transport and Distribution
    # Price|Final Energy|Industry|Liquids|Biomass|Other Taxes

    var["Price|Final Energy|Industry|Solids|Biomass"] = \
        price_load(n, "solid biomass for industry", region)[0] / (MWh2GJ)

    # Price|Final Energy|Industry|Solids|Biomass|Sales Margin x
    # Price|Final Energy|Industry|Solids|Biomass|Transport and Distribution
    # Price|Final Energy|Industry|Solids|Biomass|Other Taxes

    # Price|Final Energy|Industry|Gases|Biomass ?
    # Price|Final Energy|Industry|Gases|Biomass|Sales Margin
    # Price|Final Energy|Industry|Gases|Biomass|Transport and Distribution
    # Price|Final Energy|Industry|Gases|Biomass|Other Taxes

    var["Price|Final Energy|Industry|Liquids|Efuel"] = \
        var["Price|Final Energy|Industry|Liquids"]
    
    # Price|Final Energy|Industry|Liquids|Efuel|Sales Margin x
    # Price|Final Energy|Industry|Liquids|Efuel|Transport and Distribution
    # Price|Final Energy|Industry|Liquids|Efuel|Other Taxes

    # Price|Final Energy|Industry|Gases|Efuel x
    # Price|Final Energy|Industry|Gases|Efuel|Sales Margin
    # Price|Final Energy|Industry|Gases|Efuel|Transport and Distribution
    # Price|Final Energy|Industry|Gases|Efuel|Other Taxes

    # Price|Final Energy|Industry|Hydrogen|Sales Margin x
    # Price|Final Energy|Industry|Hydrogen|Transport and Distribution
    # Price|Final Energy|Industry|Hydrogen|Other Taxes

    # Price|Final Energy|Industry|Electricity|Sales Margin x
    # Price|Final Energy|Industry|Electricity|Transport and Distribution
    # Price|Final Energy|Industry|Electricity|Other Taxes

    # Rest Tier3
    nodal_flows_gas = get_nodal_flows(
        n, "gas", region,
        query = "not carrier.str.contains('pipeline')"
                "& not carrier == 'gas'"
                "& not carrier.str.contains('rural')"
                "& not carrier.str.contains('urban decentral')"
            )
    nodal_prices_gas = n.buses_t.marginal_price[nodal_flows_gas.columns]

    var["Price|Secondary Energy|Gases"] = \
    nodal_flows_gas.mul(nodal_prices_gas).values.sum() / nodal_flows_gas.values.sum() /MWh2GJ

    nodal_flows_oil = get_nodal_flows(
        n, "oil", region,
        query = "not carrier.str.contains('rural')"
                "& not carrier == 'oil'"
                "& not carrier.str.contains('urban decentral')"
            )
    nodal_prices_oil = n.buses_t.marginal_price[nodal_flows_oil.columns]

    var["Price|Secondary Energy|Liquids"] = \
    nodal_flows_oil.mul(nodal_prices_oil).values.sum() / nodal_flows_oil.values.sum() /MWh2GJ

    # Price|Final Energy|Transportation|Freight|Electricity x
    # Price|Final Energy|Transportation|Freight|Gases
    # Price|Final Energy|Transportation|Freight|Hydrogen

    var["Price|Final Energy|Transportation|Freight|Liquids"] = \
            var["Price|Final Energy|Transportation|Liquids|Efuel"]

    # Price|Final Energy|Transportation|Freight|Solids x

    var["Price|Final Energy|Transportation|Passenger|Electricity"] = \
        var["Price|Final Energy|Transportation|Electricity"]
    
    # Price|Final Energy|Transportation|Passenger|Gases
    # Price|Final Energy|Transportation|Passenger|Hydrogen
    var["Price|Final Energy|Transportation|Passenger|Liquids"] = \
        var["Price|Final Energy|Transportation|Liquids|Efuel"]
    
    # Price|Final Energy|Transportation|Passenger|Solids x

    # Price|Final Energy|Residential|Hydrogen x
    # Price|Final Energy|Residential|Gases|Natural Gas ?
    # Price|Final Energy|Residential|Solids|Coal x

    # Price|Final Energy|Transportation|Electricity|Carbon Price Component ?
    # Price|Final Energy|Transportation|Gases|Carbon Price Component
    # Price|Final Energy|Transportation|Hydrogen|Carbon Price Component
    # Price|Final Energy|Transportation|Liquids|Carbon Price Component

    return var

def get_investments(n_pre, n, costs, region, dg_cost_factor=1.0, length_factor=1.0):
    kwargs = {
        'groupby': n_pre.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }
    var = pd.Series()

    # capacities_electricity = n.statistics.expanded_capacity(
    #     bus_carrier=["AC", "low voltage"],
    #     **kwargs,
    # ).filter(like=region).groupby("carrier").sum() # in bn 
    # 
    # var["Investment"] = 
    # var["Investment|Energy Supply"] = \ 
    #var["Investment|Energy Supply|Electricity"] = \
    #    capex_electricity.sum()
    #var["Investment|Energy Supply|Electricity|Coal"] = \
    #    capex_electricity.reindex(["coal", "lignite"]).sum()
    # var["Investment|Energy Supply|Electricity|Coal|w/ CCS"] = \ 
    # var["Investment|Energy Supply|Electricity|Coal|w/o CCS"] = \
    # var["Investment|Energy Supply|Electricity|Gas"] = \ 
    # var["Investment|Energy Supply|Electricity|Gas|w/ CCS"] = \  
    # var["Investment|Energy Supply|Electricity|Gas|w/o CCS"] = \ 
    # var["Investment|Energy Supply|Electricity|Oil"] = \ 
    # var["Investment|Energy Supply|Electricity|Oil|w/ CCS"] = \  
    # var["Investment|Energy Supply|Electricity|Oil|w/o CCS"] = \ 
    # var["Investment|Energy Supply|Electricity|Non-fossil"] = \  
    # var["Investment|Energy Supply|Electricity|Biomass"] = \ 
    # var["Investment|Energy Supply|Electricity|Biomass|w/ CCS"] = \  
    # var["Investment|Energy Supply|Electricity|Biomass|w/o CCS"] = \ 
    # var["Investment|Energy Supply|Electricity|Nuclear"] = \ 
    # var["Investment|Energy Supply|Electricity|Non-Biomass Renewables"] = \  
    # var["Investment|Energy Supply|Electricity|Hydro"] = 
    # var["Investment|Energy Supply|Electricity|Solar"] = 
    # var["Investment|Energy Supply|Electricity|Wind"] = \
    # var["Investment|Energy Supply|Electricity|Geothermal"] = \  
    # var["Investment|Energy Supply|Electricity|Ocean"] = 
    # var["Investment|Energy Supply|Electricity|Other"] = 

    dc_links = n.links[
        (n.links.carrier=="DC") & 
        (n.links.bus0 + n.links.bus1).str.contains(region) & 
        ~n.links.index.str.contains("reversed")
    ]
    dc_expansion = dc_links.p_nom_opt - n_pre.links.loc[dc_links.index].p_nom_min

    dc_new = (dc_expansion > 10) & (n_pre.links.loc[dc_links.index].p_nom_min < 10)

    dc_investments = dc_links.length * length_factor * (
        (1 - dc_links.underwater_fraction) 
        * dc_expansion 
        * costs.at["HVDC overhead", "investment"]
        + 
        dc_links.underwater_fraction 
        * dc_expansion 
        * costs.at["HVDC submarine", "investment"] 
        
    ) + dc_new * costs.at["HVDC inverter pair","investment"] 

    ac_lines = n.lines[(n.lines.bus0 + n.lines.bus1).str.contains(region)]
    ac_expansion = ac_lines.s_nom_opt - n_pre.lines.loc[ac_lines.index].s_nom_min
    ac_investments = ac_lines.length * length_factor *  ac_expansion * costs.at["HVAC overhead", "investment"]
    var["Investment|Energy Supply|Electricity|Transmission|AC"] = \
        ac_investments.sum()    
    var["Investment|Energy Supply|Electricity|Transmission|DC"] = \
        dc_investments.sum() 
    
    var["Investment|Energy Supply|Electricity|Transmission"] = \
    var["Investment|Energy Supply|Electricity|Transmission|AC"] + \
    var["Investment|Energy Supply|Electricity|Transmission|DC"] 

    distribution_grid = n.links[
        n.links.carrier.str.contains("distribution")].filter(like="DE",axis=0)

    year = distribution_grid.build_year.max()
    year_pre = (year - 5) if year > 2020 else 2020

    dg_expansion = (
        distribution_grid.p_nom_opt.sum() 
        - distribution_grid[distribution_grid.build_year <= year_pre].p_nom_opt.sum()
    )
    dg_investment = (
        dg_expansion 
        * costs.at["electricity distribution grid", "investment"]
        * dg_cost_factor
    )
    var["Investment|Energy Supply|Electricity|Distribution"] = \
        dg_investment
    
    var["Investment|Energy Supply|Electricity|Transmission and Distribution"] = \
        var["Investment|Energy Supply|Electricity|Distribution"] + \
        var["Investment|Energy Supply|Electricity|Transmission"]
    

    h2_links = n.links[
        n.links.carrier.str.contains("H2 pipeline")
        & ~n.links.reversed
        & (n.links.bus0 + n.links.bus1).str.contains(region)
    ]
    year = n.links.build_year.max()
    new_h2_links = h2_links[
        ((year - 5) < h2_links.build_year) 
        & ( h2_links.build_year <= year)]
    h2_costs = (
        new_h2_links.length * new_h2_links.p_nom_opt 
        * costs.at["H2 pipeline", "investment"]
    )

    var["Investment|Energy Supply|Hydrogen|Transmission"] = \
        h2_costs.sum()


    gas_links = n.links[
        (
            ((n.links.carrier == "gas pipeline") & (n.links.build_year > 2020)) 
            | (n.links.carrier == "gas pipeline new")
        )
        & ~n.links.reversed
        & (n.links.bus0 + n.links.bus1).str.contains(region)
    ]
    year = n.links.build_year.max()
    new_gas_links = gas_links[
        ((year - 5) < gas_links.build_year) 
        & (gas_links.build_year <= year)]
    gas_costs = (
        new_gas_links.length * new_gas_links.p_nom_opt 
        * costs.at["CH4 (g) pipeline", "investment"]
    )

    var["Investment|Energy Supply|Gas|Transmission"] = \
        gas_costs.sum()

    # var["Investment|Energy Supply|Electricity|Electricity Storage"] = \ 
    # var["Investment|Energy Supply|Hydrogen|Fossil"] = \ 
    # var["Investment|Energy Supply|Hydrogen|Biomass"] = \
    # var["Investment|Energy Supply|Hydrogen|Electrolysis"] = 
    # var["Investment|Energy Supply|Hydrogen|Other"] = \  
    # var["Investment|Energy Supply|Liquids"] = \ 
    # var["Investment|Energy Supply|Liquids|Oil"] = \ 
    # var["Investment|Energy Supply|Liquids|Coal and Gas"] = \
    # var["Investment|Energy Supply|Liquids|Biomass"] = \ 
    # var["Investment|Energy Supply|CO2 Transport and Storage"] = 
    # var["Investment|Energy Supply|Other"] = 
    # var["Investment|Energy Efficiency"] = \ 
    # var["Investment|Energy Supply|Heat"] = \
    # var["Investment|Energy Supply|Hydrogen"] = \
    # var["Investment|RnD|Energy Supply"] = \ 
    # var["Investment|Energy Demand|Transportation"] = \  
    # var["Investment|Energy Demand|Transportation|LDV"] = \  
    # var["Investment|Energy Demand|Transportation|Bus"] = \  
    # var["Investment|Energy Demand|Transportation|Rail"] = \ 
    # var["Investment|Energy Demand|Transportation|Truck"] = \
    # var["Investment|Infrastructure|Transport"] = \  
    # var["Investment|Energy Demand|Residential and Commercial"] = \  
    # var["Investment|Energy Demand|Residential and Commercial|Low-Efficiency Buildings"] = \ 
    # var["Investment|Energy Demand|Residential and Commercial|Medium-Efficiency Buildings"] = \  
    # var["Investment|Energy Demand|Residential and Commercial|High-Efficiency Buildings"] = \
    # var["Investment|Energy Demand|Residential and Commercial|Building Retrofits"] = 
    # var["Investment|Energy Demand|Residential and Commercial|Space Heating"] = \
    # var["Investment|Infrastructure|Industry|Green"] = \ 
    # var["Investment|Infrastructure|Industry|Non-Green"] = \ 
    # var["Investment|Industry"] = \  
    return var

def get_ariadne_var(n, industry_demand, energy_totals, costs, region):

    var = pd.concat([
        get_capacities(n, region),
        get_capacity_additions_simple(n,region),
        #get_installed_capacities(n,region),
        #get_capacity_additions(n, region),
        #get_capacity_additions_nstat(n, region),
        get_primary_energy(n, region),
        get_secondary_energy(n, region),
        get_final_energy(n, region, industry_demand, energy_totals),
        get_prices(n,region), 
        get_emissions(n, region, energy_totals),
        get_investments(
            n, n, costs, region,
            dg_cost_factor=snakemake.params.dg_cost_factor,
            length_factor=snakemake.params.length_factor
        )
    ])

    return var


# uses the global variables model, scenario and var2unit. For now.
def get_data(
        n, industry_demand, energy_totals, costs, region,
        version="0.10", scenario="test"
    ):
    
    var = get_ariadne_var(n, industry_demand, energy_totals, costs, region)

    data = []
    for v in var.index:
        try:
            unit = var2unit[v]
        except KeyError:
            print("Warning: Variable '", v, "' not in Ariadne Database", sep="")
            unit = "NA"

        data.append([
            "PyPSA-Eur " + version, 
            scenario,
            region,
            v,
            unit,
            var[v],
        ])

    tab = pd.DataFrame(
        data, 
        columns=["Model", "Scenario", "Region", "Variable", "Unit", year]
    )

    return tab

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "export_ariadne_variables",
            simpl="",
            clusters=22,
            opts="",
            ll="vopt",
            sector_opts="None",
            #planning_horizons="2025",
            run="KN2045_Elec_v4"
        )


    config = snakemake.config
    ariadne_template = pd.read_excel(
        snakemake.input.template, sheet_name=None)
    var2unit = ariadne_template["variable_definitions"].set_index("Variable")["Unit"]
    industry_demands = [
        pd.read_csv(
            in_dem, 
            index_col="TWh/a (MtCO2/a)",
        ).multiply(TWh2PJ).rename_axis("bus")
        for in_dem in snakemake.input.industry_demands
    ]
    energy_totals = pd.read_csv(
        snakemake.input.energy_totals,
        index_col=[0,1],
    ).xs(
        snakemake.params.energy_totals_year,
        level="year",
    ).multiply(TWh2PJ)

    nhours = int(snakemake.params.hours[0:3])
    nyears = nhours / 8760

    costs = list(map(
        lambda _costs: prepare_costs(
            _costs,
            snakemake.params.costs,
            nyears,
        ).multiply(1e-9), # in bn €
        snakemake.input.costs
    ))


    networks = [pypsa.Network(n) for n in snakemake.input.networks]

    yearly_dfs = []
    for i, year in enumerate(snakemake.params.planning_horizons):
        yearly_dfs.append(get_data(
            networks[i],
            industry_demands[i],
            energy_totals,
            costs[i],
            "DE",
            version=config["version"],
            scenario=config["run"]["name"][0],
        ))

    df = reduce(
        lambda left, right: pd.merge(
            left, 
            right, 
            on=["Model", "Scenario", "Region", "Variable", "Unit"]), 
        yearly_dfs
    )

    df["Region"] = df["Region"].str.replace("DE", "DEU")
    df["Model"] = "PyPSA-Eur v0.10"
    print(
        "Dropping variables which are not in the template:",
        *df.loc[df["Unit"] == "NA"]["Variable"],
        sep="\n"
    )
    #df.drop(df.loc[df["Unit"] == "NA"].index, inplace=True)

    meta = pd.Series({
        'Model': "PyPSA-Eur v0.10", 
        'Scenario': snakemake.config["iiasa_database"]["reference_scenario"], 
        'Quality Assessment': "preliminary",
        'Internal usage within Kopernikus AG Szenarien': "no",
        'Release for publication': "no",
    })

    with pd.ExcelWriter(snakemake.output.exported_variables) as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        meta.to_frame().T.to_excel(writer, sheet_name="meta", index=False)


    # For debugging
    n = networks[0]
    c = costs[0]
    region="DE"
    kwargs = {
        'groupby': n.statistics.groupers.get_name_bus_and_carrier,
        'nice_names': False,
    }
    dg_cost_factor=snakemake.params.dg_cost_factor