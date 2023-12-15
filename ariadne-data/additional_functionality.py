
electrolyser_targets = {2020: 0,
                        2030: 10,
                        2040: 50,
                        2050: 80}



def add_electrolysers_for_de(n, snapshots, investment_year):

    print(f"Adding constraint on electrolyser capacity to be greater than {electrolyser_targets[investment_year]} GW")

    electrolyser_existing_index = n.links.index[(n.links.index.str[:2] == "DE") & (n.links.carrier == "H2 Electrolysis") & ~n.links.p_nom_extendable]
    electrolyser_extendable_index = n.links.index[(n.links.index.str[:2] == "DE") & (n.links.carrier == "H2 Electrolysis") & n.links.p_nom_extendable]

    existing_electrolyser_capacity = n.links.loc[electrolyser_existing_index, "p_nom"].sum()

    print(f"Existing electrolyser capacity: {existing_electrolyser_capacity} MW")

    electrolyser_pnom = n.model["Link-p_nom"].loc[electrolyser_extendable_index]

    lhs = electrolyser_pnom.sum()

    c = n.model.add_constraints(
        lhs >= 1e3*electrolyser_targets[investment_year] - existing_electrolyser_capacity, name=f"GlobalConstraint-electrolyser_capacity"
    )



def additional_functionality(n, snapshots, planning_horizons):
    print("Adding Ariadne-specific functionality")

    investment_year = int(planning_horizons[-4:])

    add_electrolysers_for_de(n, snapshots, investment_year)
