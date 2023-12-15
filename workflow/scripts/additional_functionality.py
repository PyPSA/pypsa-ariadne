
def add_min_limits(n, snapshots, investment_year, config):

    print(config["limits_min"])

    for c in n.iterate_components(config["limits_min"]):
        print(f"Adding minimum constraints for {c.list_name}")

        for carrier in config["limits_min"][c.name]:

            for ct in config["limits_min"][c.name][carrier]:
                limit = 1e3*config["limits_min"][c.name][carrier][ct][investment_year]

                print(f"Adding constraint on {c.name} {carrier} capacity in {ct} to be greater than {limit} MW")

                existing_index = c.df.index[(c.df.index.str[:2] == ct) & (c.df.carrier.str[:len(carrier)] == carrier) & ~c.df.p_nom_extendable]
                extendable_index = c.df.index[(c.df.index.str[:2] == ct) & (c.df.carrier.str[:len(carrier)] == carrier) & c.df.p_nom_extendable]

                existing_capacity = c.df.loc[existing_index, "p_nom"].sum()

                print(f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} MW")

                p_nom = n.model[c.name + "-p_nom"].loc[extendable_index]

                lhs = p_nom.sum()

                n.model.add_constraints(
                    lhs >= limit - existing_capacity, name=f"GlobalConstraint-{c.name}-{carrier.replace(' ','-')}-capacity"
                )



def additional_functionality(n, snapshots, wildcards, config):
    print("Adding Ariadne-specific functionality")

    investment_year = int(wildcards.planning_horizons[-4:])

    add_min_limits(n, snapshots, investment_year, config)
