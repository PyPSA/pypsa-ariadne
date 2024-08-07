
import pypsa

# from PyPSA-Eur solve_network.py
def add_co2_atmosphere_constraint(n):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last_i = n.snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")

n = pypsa.Network(snakemake.input.network)

n.optimize.create_model()

add_co2_atmosphere_constraint(n)

status, termination_condition = n.optimize.solve_model(solver_name=snakemake.config["solving"]["solver"]["name"],
                                                       solver_options=snakemake.config["solving"]["solver_options"]["gurobi-default"])


german_lines = n.lines.index[n.lines.bus0.str.startswith("DE") & n.lines.bus1.str.startswith("DE")]
expansion = ((n.lines.s_nom_opt - n.lines.s_nom)*n.lines.length).loc[german_lines].sum()/1e6

print(n.objective/1e9,"bnEUR/a")
print(-n.global_constraints.at["CO2Limit","mu"],"EUR/tCO2")
print(expansion,"TWkm")

n.export_to_netcdf(snakemake.output.network)
