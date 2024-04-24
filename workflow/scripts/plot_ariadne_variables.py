import pandas as pd
import matplotlib.pyplot as plt
import pyam
import os

def ariadne_subplot(
    df, ax, title, 
    select_regex="", drop_regex="", stacked=True,
):
    df = df.T.copy()
    if select_regex:
        df = df.filter(
            regex=select_regex,
        )
    if drop_regex:
        df = df.filter(
            regex=drop_regex,
        )
    # Check that all values have the same Unit
    assert df.columns.unique(level="Unit").size == 1

    # Simplify variable names
    df.columns = pd.Index(
        map(
            lambda x: x[0][(x[0].find("|") + 1):], 
            df.columns,
        ),
        name=df.columns.names[0],
    )

    return df.plot.area(ax=ax, title=title, legend=False, stacked=stacked)



def side_by_side_plot(
        df, dfhybrid, title, savepath,
        rshift=1.25, **kwargs
    ):
    idx = df.index.intersection(dfhybrid.index)
    df = df.loc[idx]
    dfhybrid = dfhybrid.loc[idx]

    fig, axes = plt.subplots(ncols=2, sharey=True)
    ax = ariadne_subplot(df, axes[0], "PyPSA-Eur", **kwargs)
    ax2 = ariadne_subplot(dfhybrid, axes[1], "REMIND-EU v1.1", **kwargs)
    
    handles, labels = ax.get_legend_handles_labels()
    labels2 = ax2.get_legend_handles_labels()[1]
    assert labels == labels2

    fig.legend(
        reversed(handles), 
        reversed(labels), 
        bbox_to_anchor=(rshift,0.9)
    )
    fig.suptitle(title)
    title = title.replace(" ", "_")
    fig.savefig(savepath, bbox_inches="tight")
    return fig 

def within_plot(df, df2, 
                title, savepath, 
                select_regex="", drop_regex="",
                unit = "EUR_2020/GJ", **kwargs
                ):

    df = df.T.copy()
    if select_regex:
        df = df.filter(
            regex=select_regex,
        )
    if drop_regex:
        df = df.filter(
            regex=drop_regex,
        )
    
    n = df.shape[1]
    if n == 0:
        print(f"Warning! Apparently the variables required for this plot (({title}) are missing.")
        fig = plt.figure()
        plt.title(f"Warning! Apparently the variables required for this plot ({title}) are missing.")
        fig.savefig(savepath, bbox_inches="tight")
        return fig
    rows = n // 2 + n % 2 

    fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
    axes = axes.flatten()

    for i, var in enumerate(df.columns.get_level_values("Variable")):

        axes[i].plot(df.xs(var, axis=1, level=0), label="PyPSA-Eur")
        if var in df2.index.get_level_values("Variable"):
            axes[i].plot(df2.T.xs(var, axis=1, level=0), label="REMIND-EU")   
        axes[i].set_title(var)
        axes[i].legend()

    # Remove the last subplot if there's an odd number of plots
    if n % 2 != 0:
        fig.delaxes(axes[-1])

    plt.suptitle(f"{title} in ({unit})", fontsize="xx-large", y=1.0)
    plt.tight_layout()
    plt.close()
    fig.savefig(savepath, bbox_inches="tight")

    return fig 

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_ariadne_variables",
            simpl="",
            clusters=22,
            opts="",
            ll="v1.2",
            sector_opts="None",
            planning_horizons="2050",
            run="KN2045_H2_v4"
        )

    df = pd.read_excel(
        snakemake.input.exported_variables,
        index_col=list(range(5)),
        #index_col=["Model", "Scenario", "Region", "Variable", "Unit"],
        sheet_name="data"
    ).groupby(["Variable","Unit"]).sum()

    df.columns = df.columns.astype(str)
    leitmodell="REMIND-EU v1.1"

    dfremind = pd.read_csv(
        snakemake.input.ariadne_database,
        index_col=["model", "scenario", "region", "variable", "unit"]
    ).loc[
        leitmodell, snakemake.params.iiasa_scenario, "Deutschland"
    ][df.columns]
    dfremind.index.names = df.index.names


    idx = df.index.intersection(dfremind.index)
    print(
        f"Dropping variables missing in {leitmodell}:", 
        df.index.difference(dfremind.index),
    )
    df = df.loc[idx]
    dfremind = dfremind.loc[idx]

    side_by_side_plot(
        df,
        dfremind,
        "Primary Energy in PJ_yr",
        savepath=snakemake.output.primary_energy,
        select_regex="Primary Energy\|[^|]*$",
        drop_regex="^(?!.*(Fossil|Price)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Primary Energy in PJ_yr",
        savepath=snakemake.output.primary_energy_detailed,
        select_regex="Primary Energy\|[^|]*\|[^|]*$",
        drop_regex="^(?!.*(CCS|Price)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Secondary Energy in PJ_yr",
        savepath=snakemake.output.secondary_energy,
        select_regex="Secondary Energy\|[^|]*$",
        drop_regex="^(?!.*(Price)).+"

    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Secondary Energy in PJ_yr",
        savepath=snakemake.output.secondary_energy_detailed,
        # Secondary Energy|Something|Something (exactly two pipes)
        select_regex="Secondary Energy\|[^|]*\|[^|]*$",
        # Not ending in Fossil or Renewables (i.e., categories)
        drop_regex="^(?!.*(Fossil|Renewables|Losses|Price)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Final Energy in PJ_yr",
        savepath=snakemake.output.final_energy,
        select_regex="Final Energy\|[^|]*$",
        drop_regex="^(?!.*(Electricity|Price)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Final Energy in PJ_yr",
        savepath=snakemake.output.final_energy_detailed,
        select_regex="Final Energy\|[^|]*\|[^|]*$",
        rshift = 1.45,
        drop_regex="^(?!.*(Price)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Capacity in GW",
        savepath=snakemake.output.capacity,
        select_regex="Capacity\|[^|]*$",
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Capacity in GW",
        savepath=snakemake.output.capacity_detailed,
        select_regex="Capacity\|[^|]*\|[^|]*$",
        drop_regex="^(?!.*(Reservoir|Converter)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Demand Emissions in Mt",
        savepath=snakemake.output.energy_demand_emissions,
        select_regex="Emissions\|CO2\|Energy\|Demand\|[^|]*$",
        stacked=False,
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Supply Emissions in Mt",
        savepath=snakemake.output.energy_supply_emissions,
        select_regex="Emissions\|CO2\|Energy\|Supply\|[^|]*$",
        stacked=False,

        drop_regex="^(?!.*(and)).+"
    )

    side_by_side_plot(
        df,
        dfremind,
        "Detailed Emissions in Mt",
        savepath=snakemake.output.co2_emissions,
        select_regex="Emissions\|CO2\|[^|]*$",
        stacked=False,
        #drop_regex="^(?!.*(and)).+"
    )

    within_plot(
        df, 
        dfremind, 
        title = "Price|Primary Energy", 
        savepath=snakemake.output.primary_energy_price,
        select_regex="Price\|Primary Energy\|[^|]*$"
    )
    
    within_plot(
        df[df.index.get_level_values("Variable").str.startswith("Price|Secondary Energy")], 
        dfremind, 
        title = "Price|Secondary Energy", 
        savepath=snakemake.output.secondary_energy_price,
    )

    # within_plot(
    #     df[df.index.get_level_values("Variable").str.startswith("Price|Final Energy|Residential")], 
    #     dfremind, 
    #     title = "Price|Final Energy|Residential", 
    #     savepath=snakemake.output.final_energy_residential_price,
    #     #select_regex="Price\|Final Energy\|Residential\|[^|]*$"
    # )

    within_plot(
        df[df.index.get_level_values("Variable").str.startswith("Price|Final Energy|Industry")], 
        dfremind, 
        title = "Price|Final Energy|Industry", 
        savepath=snakemake.output.final_energy_industry_price,
        #select_regex="Price\|Final Energy\|Industry\|[^|]*$"
    )

    within_plot(
        df[df.index.get_level_values("Variable").str.startswith("Price|Final Energy|Transportation")], 
        dfremind, 
        title = "Price|Final Energy|Transportation", 
        savepath=snakemake.output.final_energy_transportation_price,
        #select_regex="Price\|Final Energy\|Industry\|[^|]*$"
    )

    within_plot(
        df[df.index.get_level_values("Variable").str.startswith("Price|Final Energy|Residential and Commercial")], 
        dfremind, 
        title = "Price|Final Energy|Residential and Commercial", 
        savepath=snakemake.output.final_energy_residential_commercial_price,
        #select_regex="Price\|Final Energy\|Industry\|[^|]*$"
    )

    within_plot(
        df[df.index.get_level_values("Variable").str.startswith('Price')], 
        dfremind, 
        title = "All prices", 
        savepath=snakemake.output.all_prices,
    )

    within_plot(
        df[df.index.get_level_values("Variable").str.startswith('Price|Carbon')], 
        dfremind, 
        title = "Price of carbon", 
        savepath=snakemake.output.policy_carbon,
        unit="EUR/tCO2"
    )

