

import pyam

pyam.iiasa.set_config(snakemake.params.iiasa_usr, snakemake.params.iiasa_pwd)


model_raw = pyam.read_iiasa(snakemake.config["iiasa_database"]["db_name"],
                            model=snakemake.config["iiasa_database"]["model_name"],
                            scenario=snakemake.config["iiasa_database"]["scenario"])

model_df = model_raw.timeseries()

model_df = model_df.loc[snakemake.config["iiasa_database"]["model_name"],
                        snakemake.config["iiasa_database"]["scenario"],
                        snakemake.config["iiasa_database"]["region"]]

model_df.to_csv(snakemake.output.data)
