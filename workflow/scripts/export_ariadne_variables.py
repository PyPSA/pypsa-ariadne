import pandas as pd

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
            ll="v1.2",
            sector_opts="365H-T-H-B-I-A-solar+p3-linemaxext15",
            planning_horizons="2040",
        )


print(snakemake.input.networks)
df=pd.DataFrame()
df.to_csv(
    snakemake.output.ariadne_variables,
    index=False
)