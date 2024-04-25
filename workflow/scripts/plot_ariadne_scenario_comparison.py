
import pandas as pd
import matplotlib.pyplot as plt


def scenario_plot(df, var):
    unit = df._get_label_or_level_values("Unit")[0]
    df = df.droplevel("Unit")
    ax = df.T.plot(
        xlabel="years",
        ylabel=str(unit),
        title=str(var)
    )
    prefix=snakemake.config["run"]["prefix"]
    var=var.replace("|","-").replace("\\","-").replace(" ","-").replace("/","-")
    ax.figure.savefig(
        f"results/{prefix}/ariadne_comparison/{var}")

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_ariadne_scenario_comparison",
            # simpl="",
            # clusters=22,
            # opts="",
            # ll="vopt",
            # sector_opts="None",
            # planning_horizons="2050",
            # run="KN2045_Bal_v4"
        )

    dfs = []
    for file in snakemake.input:
        _df = pd.read_excel(file,
            index_col=list(range(5)),
            sheet_name="data").droplevel(["Model", "Region"])
        dfs.append(_df)

    df = pd.concat(dfs, axis=0)


    prefix=snakemake.config["run"]["prefix"]
    if not os.path.exists(f"results/{prefix}/ariadne_comparison/"):
        os.mkdir(f"results/{prefix}/ariadne_comparison/")

    for var in df._get_label_or_level_values("Variable"):
        scenario_plot(df.xs(var, level="Variable"), var)
