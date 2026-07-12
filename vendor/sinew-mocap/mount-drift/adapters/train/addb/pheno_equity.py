# SPDX-License-Identifier: MIT
# Google-style equity weighting for the phenotype (minimum-representation floors, not diversity,
# not population-proportional).
#
# Google's on-body heart-rate work guaranteed each skin-tone group a floor of the data — light and
# medium >= 25%, dark >= 33% — giving the HARDEST group (dark skin, worst for optical sensors) the
# LARGEST floor so the model performs equitably (docs/references.bib schumann2023skintone).  The IMU
# analog: sensor-to-bone calibration is hardest at high BMI, where soft-tissue artifact moves the
# mount most, so the obese band gets the largest floor; sex gets an equal floor because the salvaged
# data is otherwise ~1 woman in 60.  Iterative proportional fitting (raking) sets per-subject weights
# so the weighted sex marginal and BMI-band marginal both hit these floor-derived targets; window
# sampling then draws each group at its guaranteed rate.  Strata with no subject cannot be raked up
# and are reported as equity gaps for the ongoing download to fill.
#   python pheno_equity.py <pheno_probe.json> <out weights.json>
import sys, json
import numpy as np

POP = {"male", "female"}
# Floor-derived targets (normalised to sum 1).  Sex: equal floor.  BMI: hardest band (obese) largest,
# mirroring Google's dark-skin >= 33% (25/25/33 -> 0.30/0.30/0.40).
SEX_TARGET = {"male": 0.5, "female": 0.5}
BMI_TARGET = {"normal": 0.30, "overweight": 0.30, "obese": 0.40}
BMI_EDGES = [25.0, 30.0]  # WHO: <25 normal, 25-30 overweight, >=30 obese
# CDC NHANES 2015-2018 adult means, used only to impute unknown sex.
NHANES = {"male": (176.4, 7.6, 90.6, 21.0), "female": (162.5, 7.1, 77.5, 21.0)}


def impute_sex(h_m, m_kg):
    def ll(s):
        hh0, hh1, mm0, mm1 = NHANES[s]
        return -((h_m * 100 - hh0) / hh1) ** 2 - ((m_kg - mm0) / mm1) ** 2
    return "male" if ll("male") > ll("female") else "female"


def bmi_band(h_m, m_kg):
    b = m_kg / (h_m * h_m)
    return "normal" if b < BMI_EDGES[0] else ("overweight" if b < BMI_EDGES[1] else "obese")


BMI_ORDER = ["obese", "overweight", "normal"]  # hardest -> easiest for calibration


def best_effort_bmi(present):  # cascade an empty band's floor to the nearest harder-available band
    eff = {g: BMI_TARGET[g] for g in present}
    for g in BMI_TARGET:
        if g not in present:
            gi = BMI_ORDER.index(g)
            near = min(present, key=lambda b: (abs(BMI_ORDER.index(b) - gi), BMI_ORDER.index(b)))
            eff[near] += BMI_TARGET[g]
    tot = sum(eff.values())
    return {g: v / tot for g, v in eff.items()}


def rake(subs, sex_t, bmi_t, iters=200):  # IPF: weights so sex and BMI marginals hit the targets
    w = np.ones(len(subs))
    sx = [s["sex_eff"] for s in subs]
    bm = [s["bmi_band"] for s in subs]
    for _ in range(iters):
        for key, tgt in ((sx, sex_t), (bm, bmi_t)):
            tot = w.sum()
            for g, t in tgt.items():
                idx = [i for i, k in enumerate(key) if k == g]
                cur = w[idx].sum()
                if cur > 0:
                    w[idx] *= t * tot / cur
    return w / w.mean()


def main(in_json, out_json):
    subs = json.load(open(in_json))
    for s in subs:
        s["sex_eff"] = s["sex"] if s["sex"] in POP else impute_sex(s["height"], s["mass"])
        s["bmi"] = round(s["mass"] / (s["height"] ** 2), 1)
        s["bmi_band"] = bmi_band(s["height"], s["mass"])
    present = {s["bmi_band"] for s in subs}
    bmi_eff = best_effort_bmi(present)  # cascade empty-band floors to the hardest available band
    w = rake(subs, SEX_TARGET, bmi_eff)
    for s, wi in zip(subs, w):
        s["weight"] = round(float(wi), 4)
    N = len(subs)
    eff = w.sum() ** 2 / (w ** 2).sum()
    present_sb = {(s["sex_eff"], s["bmi_band"]) for s in subs}
    gaps = [(sx, bb) for sx in POP for bb in BMI_TARGET if (sx, bb) not in present_sb]
    aw = lambda key, g: sum(s["weight"] for s in subs if s[key] == g) / N

    print(f"{N} train subjects -> best-effort Google-style equity floors via raking")
    print(f"effective sample size: {eff:.1f} (of {N}); the floors up-weight rare groups, lowering it")
    print("sex share   raw -> weighted (target):")
    for g in SEX_TARGET:
        raw = sum(1 for s in subs if s["sex_eff"] == g) / N
        print(f"  {g:6s} {raw:.2f} -> {aw('sex_eff', g):.2f}  (target {SEX_TARGET[g]:.2f})")
    print("BMI band    raw -> weighted   (ideal floor | best-effort target):")
    for g in BMI_TARGET:
        raw = sum(1 for s in subs if s["bmi_band"] == g) / N
        be = bmi_eff.get(g, 0.0)
        print(f"  {g:10s} {raw:.2f} -> {aw('bmi_band', g):.2f}   ({BMI_TARGET[g]:.2f} | {be:.2f})")
    print("equity GAPS — (sex, BMI band) with zero subjects (fill as the download grows):")
    for sx, bb in sorted(gaps):
        print(f"  {sx:6s} {bb}")
    print("most up-weighted (hardest, rarest) subjects:")
    for s in sorted(subs, key=lambda s: -s["weight"])[:8]:
        print(f"  {s['subject']:14s} {s['sex_eff']:6s} BMI {s['bmi']:4} ({s['bmi_band']:10s}) w={s['weight']}")
    json.dump(subs, open(out_json, "w"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
