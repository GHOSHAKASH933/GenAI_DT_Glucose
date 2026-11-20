"""Fill the Paper 1 template with numbers from results/paper1_results.json and build the Overleaf folder."""
import json, re, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESDIR = ROOT / "results"
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))
from common import c, m, wil, pv, dz, table, figure
import refs
try:
    from narrative_p1 import narrative
except ImportError:
    sys.exit("The manuscript text (template.tex, narrative_p1.py, narrative_adv.py) is not part of this repository. "
             "Copy the three files from the manuscript_sources archive into paper1/paper/ to rebuild the paper.")

import os
PREVIEW = "--preview" in sys.argv       # layout check while only some seeds of the extended benchmark exist; never for delivery
ADV_SRC = os.environ.get("ADV_SRC", str(RESDIR / "paper1_advanced.json"))
LAT_SRC = os.environ.get("LAT_SRC", str(RESDIR / "paper1_latency.json"))
D = json.load(open(str(RESDIR / "paper1_results.json")))
RUNS = D["runs"]
HY = json.load(open(str(RESDIR / "paper1_hybrid.json")))["runs"]
if PREVIEW:
    _have = {r["seed"] for r in json.load(open(ADV_SRC))["runs"]}
    RUNS = [r for r in RUNS if r["seed"] in _have]; HY = [r for r in HY if r["seed"] in _have]
assert [r["seed"] for r in HY] == [r["seed"] for r in RUNS], "hybrid and main runs must share seeds"
h = lambda *path: [_get(r, path) for r in HY]
B = BUILD
if PREVIEW:
    B = B.with_name("build_preview")
T = open(HERE / "template.tex").read()

g = lambda *path: [_get(r, path) for r in RUNS]


def _get(r, path):
    for p in path:
        r = r[p]
    return r


MODELS = [("persistence", "Persistence"), ("ridge", "Ridge regression"), ("mlp", "MLP"), ("gru", "GRU"),
          ("gauss_mlp", "Gaussian MLP"), ("twin", "CVAE twin")]

tok = {}
tok["n_seeds"] = str(len(RUNS))
tok["n_patients"] = "112"; tok["n_records"] = "125"; tok["n_t1"] = "12"; tok["n_t2"] = "100"
tok["n_test_pat"] = m(g("n_patients", "test"), 0)
tok["n_test_win"] = f"{int(np.mean(g('n_windows', 'test'))):,}".replace(",", "{,}")

# ---- accuracy table
rows = []
for k, lab in MODELS:
    a = lambda f: c(g("accuracy", k, f))
    rows.append(f"{lab} & {a('rmse_30')} & {a('rmse_60')} & {a('rmse_120')} & {a('mae_60')} & "
                f"{m(g('accuracy', k, 'ceg_A_60'))} & {m(g('accuracy', k, 'ceg_AB_60'))}")
a = lambda f: c(h("accuracy", f))
rows.append(f"Ridge + CVAE (hybrid twin) & {a('rmse_30')} & {a('rmse_60')} & {a('rmse_120')} & {a('mae_60')} & "
            f"{m(h('accuracy', 'ceg_A_60'))} & {m(h('accuracy', 'ceg_AB_60'))}")
tok["acc_table"] = table(
    "Point accuracy on held-out patients (mg/dL), mean over seeds with bootstrap 95\\% interval. Clarke error grid "
    "zones are for 60-minute forecasts (\\% of windows).", "tab:acc", "lcccccc",
    r"\textbf{Model} & \textbf{RMSE 30} & \textbf{RMSE 60} & \textbf{RMSE 120} & \textbf{MAE 60} & \textbf{Zone A} & \textbf{Zone A+B}",
    rows, size=r"\footnotesize")

# ---- probabilistic table
rows = []
for k, lab in (("gauss_mlp", "Gaussian MLP"), ("twin", "CVAE twin"), ("twin_latent", "CVAE twin, latent only")):
    pr = lambda f: c(g("probabilistic", k, f), 3 if "cov" in f else 1)
    conf = (f"{pr('conf_cov90_60')} & {pr('conf_width90_60')}" if "conf_cov90_60" in RUNS[0]["probabilistic"][k]
            else "-- & --")
    rows.append(f"{lab} & {pr('crps_60')} & {pr('crps_120')} & {pr('cov90_60')} & {pr('width90_60')} & {conf}")
pr = lambda f: c(h("probabilistic", f), 3 if "cov" in f else 1)
rows.append(f"Ridge + CVAE (hybrid twin) & {pr('crps_60')} & {pr('crps_120')} & {pr('cov90_60')} & {pr('width90_60')} & {pr('conf_cov90_60')} & {pr('conf_width90_60')}")
pl = lambda f: c(h("prob_latent", f), 3 if "cov" in f else 1)
rows.append(f"Hybrid twin, latent only & {pl('crps_60')} & {pl('crps_120')} & {pl('cov90_60')} & {pl('width90_60')} & -- & --")
tok["prob_table"] = table(
    "Probabilistic quality on held-out patients. CRPS in mg/dL (lower is better). Coverage and width (mg/dL) of the "
    "central 90\\% interval at 60 minutes, before and after split-conformal correction on validation patients.",
    "tab:prob", "lcccccc",
    r"& \multicolumn{2}{c}{\textbf{CRPS}} & \multicolumn{2}{c}{\textbf{Raw 90\% interval}} & \multicolumn{2}{c}{\textbf{Conformal 90\% interval}}\\ "
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\textbf{Model} & 60 min & 120 min & coverage & width & coverage & width",
    rows, size=r"\footnotesize")

# ---- hypo table
rows = []
for k, lab in MODELS:
    hz = lambda f: c(g("hypo", k, f), 3)
    br = hz("brier") if "brier" in RUNS[0]["hypo"][k] else "--"
    rows.append(f"{lab} & {hz('auroc')} & {hz('auprc')} & {br}")
hh = lambda f: c(h("hypo", f), 3)
rows.append(f"Ridge + CVAE (hybrid twin) & {hh('auroc')} & {hh('auprc')} & {hh('brier')}")
tok["hypo_table"] = table(
    f"Prediction of glucose below 70 mg/dL within the next 60 minutes on held-out patients "
    f"(event rate {m([100 * x for x in g('hypo', 'twin', 'prevalence')])}\\% of windows). The Brier score applies only to "
    "models that output a probability.", "tab:hypo", "lccc",
    r"\textbf{Model} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{Brier score}", rows)

# ---- federated table
sites = ["T1DM", "T2DM-1", "T2DM-2", "T2DM-3"]
regs = [("local_only", "Each client alone"), ("fedavg", "FedAvg"), ("fedavg_personalised", "FedAvg + client fine-tuning"),
        ("centralised", "Centralised (pooled data)")]
rows = []
for k, lab in regs:
    rows.append(lab + " & " + " & ".join(c([r["federated"][k][s] for r in RUNS]) for s in sites))
tok["fed_table"] = table(
    "Federated training across four clients that hold different diabetes types. 60-minute RMSE (mg/dL) on each "
    "client's held-out patients.", "tab:fed", "lcccc",
    r"\textbf{Regime} & \textbf{T1DM client} & \textbf{T2DM client 1} & \textbf{T2DM client 2} & \textbf{T2DM client 3}",
    rows, size=r"\footnotesize")
sz = lambda s: m([r["federated_sizes"][s] for r in RUNS], 0)
tok["fed_sizes"] = ", ".join(f"{s} {sz(s)}" for s in sites)

# ---- edge table
rows = []
for k, lab in (("twin", "CVAE twin (float32)"), ("twin_int8", "CVAE twin (int8)"), ("student", "Distilled student")):
    rows.append(f"{lab} & {m(g('edge', k, 'kb'), 1)} & {c(g('edge', k, 'ms'), 3)} & {c(g('edge', k, 'rmse_60'))}")
tok["edge_table"] = table(
    "Edge deployment. Serialised size, median latency for one window on one CPU thread (the twin draws 32 samples), "
    "and 60-minute RMSE on held-out patients.", "tab:edge", "lccc",
    r"\textbf{Model} & \textbf{Size (kB)} & \textbf{Latency (ms)} & \textbf{RMSE 60 (mg/dL)}", rows)

def wide(t):
    return t.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}", 1).replace(
        "\\end{table}", "\\end{adjustwidth}\n\\end{table}", 1)
for k in ("acc_table", "prob_table", "fed_table"):
    tok[k] = wide(tok[k])

# ---- figures
tok["fig_horizon"] = figure("fig2_accuracy.png", "(a) RMSE of each model at 30, 60 and 120 minutes, mean over seeds. "
                            "(b) Share of 60-minute forecasts in zone A of the Clarke error grid, with bootstrap 95\\% intervals.",
                            "fig:acc")
tok["fig_scen"] = figure("fig3_scenarios.png", "Twenty sampled two-hour scenarios for four held-out windows (two type 1, two "
                         "type 2), seed 0. Solid black: the two hours of history; dashed black: what actually happened. Top and "
                         "middle: the hybrid and plain twins, sampling only the latent variable. Bottom: the Gaussian MLP, whose "
                         "samples are independent from step to step. The dotted line marks 70 mg/dL. The windows were not chosen at "
                         "random: within each diabetes type they sit at about the 75th and 88th percentile of the variability of the "
                         "observed future.", "fig:scen")
tok["fig_pers"] = figure("fig4_personalisation.png", "Adapting the twin to a new patient. 60-minute RMSE on the second half "
                         "of each held-out record after using one, two or four days from its start, for three adaptation "
                         "strategies. Error bars are bootstrap 95\\% intervals over seeds.", "fig:pers", width=r"0.62\linewidth")
tok["fig_fed"] = figure("fig5_federated.png", "60-minute RMSE on each client's held-out patients under four training "
                        "regimes.", "fig:fed", width=r"0.85\linewidth")

tok.update(narrative(RUNS, g, HY, h))

# ---- extended benchmark (p1_advanced.py): tables, figures and prose
import adv_tables as AT
from narrative_adv import adv_narrative, adv_framing   # see the note above
ADV = json.load(open(ADV_SRC))["runs"]
LAT = json.load(open(LAT_SRC))
assert [r["seed"] for r in ADV] == [r["seed"] for r in RUNS], "advanced and main runs must share seeds"
FR = [float(f) for f in ADV[0]["train_size"]]
tok["adv_acc_table"] = AT.acc_table(ADV); tok["adv_risk_table"] = AT.risk_table(ADV)
tok["sub_table"] = AT.sub_table(ADV); tok["abl_table"] = AT.abl_table(ADV)
tok["size_table"] = AT.size_table(ADV, FR, n_full=int(round(np.mean([r["n_patients"]["train"] for r in ADV]))))
tok["lat_table"] = AT.lat_table(LAT)
tok["fig_tradeoff"] = figure("fig6_tradeoff.png", "Point accuracy against scenario realism. Each marker is the mean over seeds, with "
                             "bootstrap 95\\% intervals. Filled markers use a full predictive draw; hollow squares, for the models "
                             "with a latent variable, use a draw from the latent alone. Lower is better on both axes.", "fig:tradeoff", width=r"0.8\linewidth")
tok["fig_scen_adv"] = figure("fig8_scenarios_adv.png", "Twenty sampled two-hour scenarios from five models for the same four held-out windows "
                             "as Figure~\\ref{fig:scen} (seed 0; selection as described there). Solid black: history; dashed black: what happened. "
                             "The dotted line marks 70 mg/dL.", "fig:scen_adv")
tok["fig_calib"] = figure("fig7_calibration.png", "(a) Coverage of central prediction intervals at 60 minutes before conformal "
                          "correction, averaged over seeds. (b) Reliability of the predicted probability of glucose below 70 mg/dL "
                          "within 60 minutes, pooled over seeds; bins with fewer than 30 windows are omitted.", "fig:calib")
tok["fig_abl"] = figure("fig9_ablation_size.png", "(a) 60-minute RMSE after removing input channels. (b) 60-minute RMSE when ridge "
                        "regression and the hybrid twin are trained on a fraction of the training patients. Error bars are "
                        "bootstrap 95\\% intervals over seeds.", "fig:abl")
TA = adv_narrative(ADV, RUNS, HY, LAT, g, h)
# per-type (Mondrian) conformal intervals come from a second run of p1_advanced.py on the same splits,
# which stores pooled and per-type intervals side by side
RERUN_SRC = os.environ.get("RERUN_SRC", str(RESDIR / "paper1_advanced_rerun.json"))
RR = json.load(open(RERUN_SRC))["runs"]
assert [r["seed"] for r in RR] == [r["seed"] for r in ADV], "rerun and advanced runs must share seeds"
from narrative_adv import mond_narrative
tok["mond_table"] = AT.mond_table(RR)
TA.update(mond_narrative(RR, ADV))
tok.update({k: v for k, v in TA.items() if isinstance(v, str)})
tok.update({k: v for k, v in adv_framing(ADV, RUNS, HY, LAT, g, h, TA).items() if isinstance(v, str)})

out = T
for k, v in tok.items():
    out = out.replace(f"@@{k}@@", v)
left = re.findall(r"@@[a-z0-9_]+@@", out)
out = out.replace("@@references@@", refs.bibliography(out, "mdpi"))
assert "@@references@@" not in out
if left:
    print("UNFILLED:", sorted(set(left)))

B.mkdir(parents=True, exist_ok=True)
if (HERE / "Definitions").exists():
    shutil.copytree(HERE / "Definitions", B / "Definitions", dirs_exist_ok=True)
else:
    print("NOTE: put the official MDPI Definitions folder (mdpi.cls etc.) into paper1/paper/Definitions before compiling.")
(B / "main.tex").write_text(out)
print("wrote", B / "main.tex", len(out.split()), "words (approx.)")
