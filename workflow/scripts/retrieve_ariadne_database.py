import pyam

pyam.iiasa.set_config(snakemake.params.iiasa_usr, snakemake.params.iiasa_pwd)

db = pyam.read_iiasa(
    "ariadne_intern",
    model=[# Download only the Leitmodelle
        "Hybrid", 
        "REMIND-EU v1.1", 
        'REMod v1.0', 
        'TIMES PanEU v1.0', 
        'FORECAST v1.0',
        'DEMO v1',
    ],
    scenario=[# Download only the most recent iterations of scenarios
        "8Gt_Bal_v3", 
        "8Gt_Elec_v3", 
        "8Gt_H2_v3",
    ],
)

db.timeseries().to_csv(snakemake.output.data)
