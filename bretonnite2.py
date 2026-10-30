#!/usr/bin/env python3
"""
Score de "consonance bretonne" des lieux-dits, v2.

Nouveautés par rapport à v1
---------------------------
- Lecture directe du cadastre Etalab (GeoJSON de polygones "lieux_dits", .json ou .json.gz),
  du GPKG/GeoJSON en général, et du CSV (lon/lat). Nécessite geopandas pour les formats SIG.
- n-grammes 3 à 5 caractères, min_df=20, régularisation plus forte : moins de bruit.
- Scores "out-of-fold" pour les communes d'entraînement : le Finistère et la Mayenne
  ne sont plus notés par un modèle qui les a déjà vus.
- Grille 2 km par défaut, lissage bayésien plus fort, et cellules peu fournies
  conservées avec reliable=false (à afficher en gris = "pas assez de données").
Restreignez la zone positive. Le Finistère entier inclut des communes urbaines ou francisées. Passez avec --pos-insee-file une liste de communes du Finistère ouest ou du Trégor-Cornouaille, par exemple celles où le breton est massivement attesté. Vous aurez un signal plus net.
Usage
-----
  python bretonnite2.py "data/cadastre-*-lieux_dits.json.gz" \
      --pos-depts 29 --neg-depts 53 72 50 14 61 \
      --out-prefix sortie/bzh
"""
import argparse
import glob
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

# Les articles/prépositions (le, la, les, l', du, de, des, d') sont CONSERVÉS : ils portent
# du signal ("La Goulafrière" est plus français que "Ker Mezenn"). Les articles bretons
# (an, ar, al) sont déjà gardés. L'élision l'/d' devient un mot séparé ("l", "d").
ELISION = re.compile(r"\\b([ld])['’]")


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'")
    s = re.sub(r"(?<=c) h\b", "'h", s)
    s = ELISION.sub(r"\\1 ", s)
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return " ".join(s.split())


# Chargement et traitement des lieux dits, dans un gros dataframe
def load(pattern, *_):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"Aucun fichier pour {pattern}")
    d = pd.concat([pd.read_csv(p, sep=";", dtype=str, low_memory=False) for p in paths],
                  ignore_index=True)
    d = d[d["type_voie"] == "3"]   # lieux-dits seulement
    df = pd.DataFrame({
        "nom": d["libelle"],
        "insee": d["code_dep"].str.strip().str[-2:] + d["code_commune"].str.strip().str.zfill(3),
    }).dropna()
    df["norm"] = df["nom"].map(normalize)
    df = df[df["norm"].str.len() >= 3].copy()
    df["dept"] = df["insee"].str[:2]
    return df


def build_model(min_df, C):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                          sublinear_tf=True)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    return vec, clf


def score_all(df, pos_depts, neg_depts, insee_pos, min_df, C):
    pos = df[df["insee"].isin(insee_pos)] if insee_pos else df[df["dept"].isin(pos_depts)]
    neg = df[df["dept"].isin(neg_depts)]
    tr = pd.concat([pos.assign(y=1), neg.assign(y=0)])
    tr = tr.drop_duplicates(subset=["insee", "norm"]).reset_index(drop=True)
    print(f"Entraînement : {len(pos)} positifs bruts, {len(neg)} négatifs bruts, "
          f"{len(tr)} après dédoublonnage par commune")

    vec, clf = build_model(min_df, C)
    X = vec.fit_transform(tr["norm"])

    # scores honnêtes pour les données d'entraînement : validation croisée PAR COMMUNE
    oof = cross_val_predict(clf, X, tr["y"], groups=tr["insee"],
                            cv=GroupKFold(n_splits=3), method="predict_proba")[:, 1]
    print(f"AUC validée par commune : {roc_auc_score(tr['y'], oof):.3f} "
          f"(sépare les 2 zones ; ne prouve pas la 'vraie' bretonnité)")
    tr["oof"] = oof

    clf.fit(X, tr["y"])
    df = df.copy()
    df["proba"] = clf.predict_proba(vec.transform(df["norm"]))[:, 1]
    # remplace par le score out-of-fold quand le lieu-dit était dans l'entraînement
    key = tr.set_index(["insee", "norm"])["oof"]
    idx = pd.MultiIndex.from_frame(df[["insee", "norm"]])
    oof_vals = key.reindex(idx).to_numpy()
    mask = ~np.isnan(oof_vals)
    df.loc[mask, "proba"] = oof_vals[mask]
    print(f"{mask.sum()} lieux-dits notés en out-of-fold, {(~mask).sum()} par le modèle final")
    return df, vec, clf


def top_features(vec, clf, n=50):
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("\nN-grammes les plus 'bretons' :\n", ", ".join(names[order[::-1][:n]]))
    print("\nN-grammes les plus 'non bretons' :\n", ", ".join(names[order[:n]]))
    return pd.DataFrame({"ngram": names, "coef": clf.coef_[0]}).sort_values("coef")


def by_commune(df, thr, min_n, prior_strength):
    df = df.assign(high=(df["proba"] >= thr).astype(int))
    df = df.drop_duplicates(subset=["insee", "norm"])
    g = df.groupby("insee").agg(n=("high", "size"), n_high=("high", "sum"),
                                mean_proba=("proba", "mean"))
    prior = df["high"].mean()
    g["share_raw"] = g["n_high"] / g["n"]
    g["share"] = (g["n_high"] + prior_strength * prior) / (g["n"] + prior_strength)
    g["reliable"] = g["n"] >= min_n
    g["exemples"] = (df.sort_values("proba", ascending=False).groupby("insee").head(3)
                       .groupby("insee")["nom"].apply(lambda s: " / ".join(s)))
    return g.reset_index()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--pos-depts", nargs="+", default=["29"])
    ap.add_argument("--neg-depts", nargs="+", default=["53", "72"])
    ap.add_argument("--pos-insee-file")
    ap.add_argument("--min-df", type=int, default=50)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--out-prefix", default="bretonnite")
    a = ap.parse_args()

    df = load(a.pattern)
    print(f"{len(df)} lieux-dits chargés dans {df['dept'].nunique()} départements")
    print(df.groupby("dept").agg(lieux_dits=("nom", "size"), communes=("insee", "nunique"))
            .to_string())

    # Lecture éventuelle fichier restreint positif
    insee_pos = set(open(a.pos_insee_file).read().split()) if a.pos_insee_file else None

    df, vec, clf = score_all(df, a.pos_depts, a.neg_depts, insee_pos, a.min_df, a.C)
    feats = top_features(vec, clf)

    #feats.to_csv(f"{a.out_prefix}_ngrams.csv", index=False)
    g = by_commune(df, a.thr, a.min_n, a.prior_strength)
    g.to_csv(f"{a.out_prefix}_communes.csv", index=False)
    print(f"\n{len(g)} communes, dont {int(g['reliable'].sum())} avec >= {a.min_n} lieux-dits")


if __name__ == "__main__":
    main()
