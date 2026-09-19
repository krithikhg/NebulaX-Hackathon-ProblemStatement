"""Scratch: try XGBoost on the ACV features we've found so far.

One row per (case, car). Features: everything from `car_stats` (pivoted wide
by param x statistic), plus `cooling_gap_stats`, plus the raw median indoor
temp (the "hottest car" rule) and indoor-outdoor correlation. Target:
`is_faulty` (0/1). Evaluated with **leave-one-case-out** cross-validation --
the only honest option with 6 labelled cases: train on 5, predict the 6th,
repeat for each, since a random row-level split would leak siblings from the
same file into both train and test.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import acv_features as af

labels = af.label_map()


def build_feature_table(cases):
    stats = af.all_car_stats(cases=cases)
    wide = stats.pivot_table(
        index=["case", "car"],
        columns="param",
        values=["mean", "std", "n_unique", "n_switch", "duty",
                "resid_mean", "resid_absmean", "resid_std", "resid_p95",
                "frac_majority"],
    )
    wide.columns = [f"{stat}__{param}" for stat, param in wide.columns]
    wide = wide.reset_index()

    cg = af.all_cooling_gap_stats(cases=cases)[
        ["case", "car", "cooling_gap_mean", "cooling_gap_std", "cooling_gap_max", "n_cooling"]]

    # Raw median indoor temp (the "hottest car" rule) and its within-case rank.
    hot_rows = []
    corr_rows = []
    for case in cases:
        df = af.load_case(case)
        cars = af.car_columns(df)
        for car, cols in cars.items():
            by_param = {af.canonical_param(c.split(" - ", 1)[1]): df[c] for c in cols}
            indoor = pd.to_numeric(by_param.get("Indoor Average Temperature"), errors="coerce")
            outdoor = pd.to_numeric(by_param.get("Outdoor Average Temperature"), errors="coerce")
            hot_rows.append({"case": case, "car": car,
                             "indoor_median": indoor.median() if indoor is not None else np.nan})
            corr_rows.append({"case": case, "car": car,
                              "indoor_outdoor_corr": indoor.corr(outdoor)
                              if indoor is not None and outdoor is not None else np.nan})
    hot = pd.DataFrame(hot_rows)
    hot["indoor_median_rank"] = hot.groupby("case")["indoor_median"].rank(ascending=False)
    corr = pd.DataFrame(corr_rows)

    feat = wide.merge(cg, on=["case", "car"], how="left") \
               .merge(hot, on=["case", "car"], how="left") \
               .merge(corr, on=["case", "car"], how="left")
    feat["faulty_car"] = feat["case"].map(labels)
    feat["is_faulty"] = (feat["car"] == feat["faulty_car"]).astype(int)
    return feat


def score_ranking(case_df, prob_col):
    order = list(case_df.sort_values(prob_col, ascending=False)["car"])
    faulty = case_df["faulty_car"].iloc[0]
    n = len(order)
    r = order.index(faulty) + 1
    return (n - (r - 1)) / n, r, order


def leave_one_case_out(feat, feature_cols):
    cases = feat["case"].unique()
    results = []
    for held_out in cases:
        train = feat[feat["case"] != held_out]
        test = feat[feat["case"] == held_out]

        model = XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=0,
        )
        model.fit(train[feature_cols], train["is_faulty"])
        prob = model.predict_proba(test[feature_cols])[:, 1]
        test = test.copy()
        test["prob"] = prob
        score, rank, order = score_ranking(test, "prob")
        results.append({"held_out_case": held_out, "faulty_car": test["faulty_car"].iloc[0],
                        "rank": rank, "score": score, "ranked_cars": "|".join(order)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    feat = build_feature_table(af.TRAIN_CASES)
    feature_cols = [c for c in feat.columns
                    if c not in ("case", "car", "faulty_car", "is_faulty")]
    print(f"{len(feat)} rows, {len(feature_cols)} features, "
          f"{feat['is_faulty'].sum()} positive")

    res = leave_one_case_out(feat, feature_cols)
    print(res.to_string(index=False))
    print("\nmean LOCO score:", res["score"].mean().round(3))
