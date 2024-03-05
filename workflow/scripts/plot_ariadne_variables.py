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
    ax2 = ariadne_subplot(dfhybrid, axes[1], "Hybrid", **kwargs)
    
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
            run="240219-test/normal"
        )

    
    df = pd.read_csv(
        snakemake.input.ariadne_variables,
        index_col=["Model", "Scenario", "Region", "Variable", "Unit"]
    ).groupby(["Variable","Unit"]).sum()

    df.columns = pd.to_numeric(df.columns)

    # Set USERNAME and PASSWORD for the Ariadne DB
    pyam.iiasa.set_config(
        os.environ["IIASA_USERNAME"], 
        os.environ["IIASA_PASSWORD"],
    )

    model_df= pyam.read_iiasa(
        "ariadne_intern",
        model="Hybrid",
        scenario="8Gt_Bal_v3",
    ).timeseries()

    dfhybrid = model_df.loc[
        "Hybrid", "8Gt_Bal_v3", "Deutschland"
    ][pd.to_numeric(df.keys())]
    dfhybrid.index.names = df.index.names

    side_by_side_plot(
        df,
        dfhybrid,
        "Primary Energy in PJ_yr",
        savepath=snakemake.output.primary_energy,
        select_regex="Primary Energy\|[^|]*$",
        drop_regex="^(?!.*(Fossil)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Primary Energy in PJ_yr",
        savepath=snakemake.output.primary_energy_detailed,
        select_regex="Primary Energy\|[^|]*\|[^|]*$",
        drop_regex="^(?!.*(CCS)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Secondary Energy in PJ_yr",
        savepath=snakemake.output.secondary_energy,
        select_regex="Secondary Energy\|[^|]*$",
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Secondary Energy in PJ_yr",
        savepath=snakemake.output.secondary_energy_detailed,
        # Secondary Energy|Something|Something (exactly two pipes)
        select_regex="Secondary Energy\|[^|]*\|[^|]*$",
        # Not ending in Fossil or Renewables (i.e., categories)
        drop_regex="^(?!.*(Fossil|Renewables|Losses)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Final Energy in PJ_yr",
        savepath=snakemake.output.final_energy,
        select_regex="Final Energy\|[^|]*$",
        drop_regex="^(?!.*(Electricity)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Final Energy in PJ_yr",
        savepath=snakemake.output.final_energy_detailed,
        select_regex="Final Energy\|[^|]*\|[^|]*$",
        rshift = 1.45,
        #drop_regex="^(?!.*(Electricity)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Capacity in GW",
        savepath=snakemake.output.capacity,
        select_regex="Capacity\|[^|]*$",
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Capacity in GW",
        savepath=snakemake.output.capacity_detailed,
        select_regex="Capacity\|[^|]*\|[^|]*$",
        drop_regex="^(?!.*(Reservoir|Converter)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Demand Emissions in Mt",
        savepath=snakemake.output.energy_demand_emissions,
        select_regex="Emissions\|CO2\|Energy\|Demand\|[^|]*$",
        stacked=False,
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Supply Emissions in Mt",
        savepath=snakemake.output.energy_supply_emissions,
        select_regex="Emissions\|CO2\|Energy\|Supply\|[^|]*$",
        stacked=False,
        drop_regex="^(?!.*(and)).+"
    )

    side_by_side_plot(
        df,
        dfhybrid,
        "Detailed Supply Emissions in Mt",
        savepath=snakemake.output.co2_emissions,
        select_regex="Emissions\|CO2\|[^|]*$",
        stacked=False,
        #drop_regex="^(?!.*(and)).+"
    )