

import pypsa

from pypsa.descriptors import get_switchable_as_dense as as_dense

drop_threshold = 0.001


def set_generators(zonal, redispatch):

    drop_generators = zonal.generators.index[zonal.generators.p_nom_opt < drop_threshold]
    print("dropping",drop_generators)
    zonal.generators.drop(drop_generators, inplace=True)
    redispatch.generators.drop(drop_generators, inplace=True)

    generators = n.generators.index[n.generators.bus.map(n.buses.carrier) == "AC"]
    
    redispatch.generators.loc[generators,"p_nom"] = zonal.generators.loc[generators,"p_nom_opt"]
    redispatch.generators.loc[generators,"p_nom_extendable"] = False

    p = zonal.generators_t.p[generators] / zonal.generators.loc[generators,"p_nom_opt"]
    redispatch.generators_t.p_min_pu[generators] = p
    redispatch.generators_t.p_max_pu[generators] = p


    g_up = redispatch.generators.loc[generators].copy()
    g_down = redispatch.generators.loc[generators].copy()

    g_up.index = g_up.index.map(lambda x: x + " ramp up")
    g_down.index = g_down.index.map(lambda x: x + " ramp down")

    up = (
            as_dense(zonal, "Generator", "p_max_pu")[generators] * zonal.generators.p_nom_opt.loc[generators] - zonal.generators_t.p[generators]
        ).clip(0) / zonal.generators.p_nom_opt.loc[generators]
    down = -zonal.generators_t.p[generators] / zonal.generators.p_nom_opt.loc[generators]

    up.columns = up.columns.map(lambda x: x + " ramp up")
    down.columns = down.columns.map(lambda x: x + " ramp down")

    redispatch.madd(
        "Generator",
        g_up.index,
        p_max_pu=up,
        p_min_pu=0,
        carrier=g_up.carrier + " ramp up",
        **g_up.drop(["p_min_pu","p_max_pu","carrier"], axis=1),
    )
    
    redispatch.madd(
        "Generator",
        g_down.index,
        p_min_pu=down,
        p_max_pu=0,
        carrier=g_down.carrier + " ramp down",
        **g_down.drop(["p_max_pu", "p_min_pu","carrier"], axis=1),
    )

    print(redispatch.generators_t.p_max_pu.isna().any().any())

def set_links(zonal, redispatch):

    drop_links = zonal.links.index[zonal.links.p_nom_opt < drop_threshold]
    print("dropping",drop_links)
    zonal.links.drop(drop_links, inplace=True)
    redispatch.links.drop(drop_links, inplace=True)

    links = zonal.links.index[(zonal.links.bus1.map(zonal.buses.carrier) == "AC") & (zonal.links.bus0.map(zonal.buses.carrier) != "AC")]
    links = links.append(zonal.links.index[(zonal.links.bus1.map(zonal.buses.carrier) != "AC") & (zonal.links.bus0.map(zonal.buses.carrier) == "AC")])

    redispatch.links.loc[links,"p_nom"] = zonal.links.loc[links,"p_nom_opt"]
    redispatch.links.loc[links,"p_nom_extendable"] = False

    p = zonal.links_t.p0[links] / zonal.links.loc[links,"p_nom_opt"]
    p.isna().any().any()

    (zonal.links.loc[links,"p_nom_opt"] == 0).any()

    links[zonal.links.loc[links,"p_nom_opt"] == 0]

    p = zonal.links_t.p0[links] / zonal.links.loc[links,"p_nom_opt"]
    redispatch.links_t.p_min_pu[links] = p
    redispatch.links_t.p_max_pu[links] = p

    redispatch_links = links[~zonal.links.loc[links,"carrier"].isin(["H2 Electrolysis","electricity distribution grid","DAC","battery discharger","battery charger"])]
    zonal.links.loc[redispatch_links,"carrier"].value_counts()

    g_up = redispatch.links.loc[redispatch_links].copy()
    g_down = redispatch.links.loc[redispatch_links].copy()

    g_up.index = g_up.index.map(lambda x: x + " ramp up")
    g_down.index = g_down.index.map(lambda x: x + " ramp down")

    up = (
        as_dense(zonal, "Link", "p_max_pu")[redispatch_links] * zonal.links.p_nom_opt.loc[redispatch_links] - zonal.links_t.p0[redispatch_links]
    ).clip(0) / zonal.links.p_nom_opt.loc[redispatch_links]
    down = -zonal.links_t.p0[redispatch_links] / zonal.links.p_nom_opt.loc[redispatch_links]

    up.columns = up.columns.map(lambda x: x + " ramp up")
    down.columns = down.columns.map(lambda x: x + " ramp down")

    redispatch.madd("Link",
    g_up.index,
    p_max_pu=up,
    p_min_pu=0,
        carrier=g_up.carrier + " ramp up",
    **g_up.drop(["p_min_pu","p_max_pu","carrier"], axis=1))
    redispatch.madd(
        "Link",
        g_down.index,
        p_min_pu=down,
        p_max_pu=0,
        carrier=g_up.carrier + " ramp down",
        **g_down.drop(["p_max_pu", "p_min_pu","carrier"], axis=1),
    )


def set_stores(zonal, redispatch):
    
    stores = zonal.stores.index[~zonal.stores.carrier.isin(["oil","co2","uranium","H2","gas","methanol","solid biomass","biogas","coal","lignite"])]

    redispatch.stores.loc[stores,"e_nom"] = zonal.stores.loc[stores,"e_nom_opt"]
    redispatch.stores.loc[stores,"e_nom_extendable"] = False


def add_reserve(redispatch):

    locations = redispatch.buses.index #redispatch.buses.index[redispatch.buses.carrier == "AC"] # 
    locations

    redispatch.madd("Generator",
           locations + " reserve",
           bus=locations,
           p_nom_extendable=True,
           carrier="reserve",
           capital_cost=5e5,
           marginal_cost=200)


zonal = pypsa.Network(snakemake.input.zonal)
n = pypsa.Network(snakemake.input.network)


set_generators(zonal, n)

set_links(zonal, n)

set_stores(zonal,n)


add_reserve(n)

n.export_to_netcdf(snakemake.output.network)
