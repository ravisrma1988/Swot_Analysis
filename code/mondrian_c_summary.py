"""Print the Version C group-conditional (E16 confirmation) summary."""
import pathlib
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results" / "final" / "mondrian_C"
pd.set_option("display.width", 200)
print(pd.read_csv(R / "mondrian_overall.csv").round(3).to_string(index=False))
g = pd.read_csv(R / "mondrian_groups.csv")
s = g[((g.attribute == "width") & (g.group == "<100 m")) | ((g.attribute == "latitude") & (g.group == ">=55N"))]
print(s.pivot_table(index=["type", "alpha", "attribute", "group"], columns="scheme", values="sdr").round(3).to_string())
