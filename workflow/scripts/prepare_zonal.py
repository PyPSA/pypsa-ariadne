
import pypsa, pandas as pd


def squash_network(n):
    new = n.copy()
    mapping = pd.Series(n.buses.index,n.buses.index)
    de_elec_buses = n.buses.index[n.buses.index.str.startswith("DE") & (n.buses.carrier == "AC")]

    bus = "DE1"

    mapping.loc[de_elec_buses] = bus

    new.add("Bus",
            bus,
            carrier="AC",
            x=n.buses.loc[de_elec_buses,"x"].mean(),
            y=n.buses.loc[de_elec_buses,"y"].mean(),
            )
    new.buses.at[bus,"location"] = bus
    
    print(new.buses.loc[bus])
    
    new.buses.drop(de_elec_buses,
                   inplace=True)

    for c in new.iterate_components(["Generator","Store","Load","Link","Line","StorageUnit"]):
        print(c.name)
        bus_cols = c.df.columns[c.df.columns.str.startswith("bus")]
        print(bus_cols)
        for col in bus_cols:
            c.df[col] = c.df[col].map(mapping)

    for c in new.iterate_components(["Link","Line"]):
        internal_edges = c.df.index[c.df.bus0 == c.df.bus1]
        print(c.name,internal_edges)
        c.df.drop(internal_edges,inplace=True)

    return new


original = pypsa.Network(snakemake.input.network)

n = squash_network(original)

n.export_to_netcdf(snakemake.output.network)
