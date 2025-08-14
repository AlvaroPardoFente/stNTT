import sys

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

df = pd.read_csv(sys.argv[1])

median_df = df.groupby(["N", "Block size"]).median()

print(median_df.sort_values(["N", "Block size"]))
