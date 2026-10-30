import geopandas as gpd
import pandas as pd

gdf = gpd.read_file("communes_ouest.geojson")
df = pd.read_csv("topo_communes.csv", dtype={"insee": str})

carte = gdf.merge(df, left_on="code", right_on="insee", how="left")
carte = carte.drop(columns=["insee"])
print(carte.shape, carte["mean_proba"].isna().sum(), "sans valeur")

# Simplification (en mètres, donc on passe en Lambert 93 puis retour en WGS84)
carte = carte.to_crs(2154)
carte["geometry"] = carte.geometry.simplify(100, preserve_topology=True)
carte = carte.to_crs(4326)

carte["mean_proba"] = carte["mean_proba"].round(4)
carte.to_file("carte.geojson", driver="GeoJSON")