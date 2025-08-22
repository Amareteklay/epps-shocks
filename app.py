from __future__ import annotations
import os, glob, shutil, subprocess
from pathlib import Path

import streamlit as st
import pandas as pd

from epps_shocks.modeling_grid import generate_model_grid, write_grid_batches, filter_pending_models
from epps_shocks.merge_results import merge_model_results, rank_models
from epps_shocks.config import DATA_RAW, DATA_INTERIM, MAX_LAG
from epps_shocks.prep import build_and_save
from epps_shocks.features import build_full_panel


st.set_page_config(page_title="Shocks & EPPs – Modeling")


def find_rscript(user_path = None):
    
    if user_path:
        p = Path(user_path.strip().strip('"'))
        return str(p) if p.exists() else None

    env_p = os.environ.get("RSCRIPT")
    if env_p and Path(env_p).exists():
        return env_p

    which = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if which and Path(which).exists():
        return which

    candidates: list[str] = []
    for base in (r"C:\Program Files\R", r"C:\Program Files (x86)\R"):
        for d in glob.glob(os.path.join(base, "R-*")):
            p = Path(d) / "bin" / "Rscript.exe"
            if p.exists():
                candidates.append(str(p))
    if candidates:
        # newest by folder name
        return sorted(candidates)[-1]

    return None


def get_project_root(start):
    
    cur = start
    target = Path("r") / "run_grid.R"
    for _ in range(6):  # don't traverse too far
        if (cur / target).exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent


st.title("Data Overview")
st.subheader("Raw data")
don_path    = DATA_RAW / "DONdatabase.csv"
shocks_path = DATA_RAW / "Shocks_Database_counts.csv"

don_out, shocks_out = build_and_save(don_path, shocks_path)
st.success("Processed files saved")

don_df    = pd.read_csv(don_out)
shocks_df = pd.read_csv(shocks_out)


st.subheader("Preprocessed data")
shocks_file = DATA_INTERIM / "shocks_processed.csv"
don_file    = DATA_INTERIM / "don_processed.csv"

shocks_df = pd.read_csv(shocks_file)
don_df    = pd.read_csv(don_file)

panel = build_full_panel(shocks_df=shocks_df, don_df=don_df, max_lag=int(MAX_LAG))
panel_out = Path("data/03_processed/full_panel.csv")
panel_out.parent.mkdir(parents=True, exist_ok=True)
panel.to_csv(panel_out, index=False)

st.subheader("Preview")
st.caption(f"Rows: {len(panel):,} • Columns: {panel.shape[1]}")
st.dataframe(panel.head(100), use_container_width=True)

st.header("Modeling")

panel_path  = st.text_input("Panel CSV path", value=str(panel_out))
results_dir = st.text_input("Results dir", value="results/lags")
merged_out  = st.text_input("Merged output file", value="results/summaries/model_results_lags.parquet")
specs_dir   = st.text_input("Specs dir", value="specs")
batch_size  = st.number_input("Batch size", min_value=100, max_value=5000, value=1000, step=100)
engine      = st.selectbox("R engine", options=["glmmTMB","glmer"], index=0)

with st.expander("Grid options", expanded=True):
    preds_raw = st.text_area("Candidate predictors (one per line)",
                             value="\n".join(["Year","outbreak_lag"]))
    predictors = [p.strip() for p in preds_raw.splitlines() if p.strip()]

    scopes = st.multiselect("Scopes", ["Global","Africa","Asia","Europe","America"],
                            default=["Global","Africa","Asia","Europe","America"])
    fe_by_scope   = {s: [""] for s in scopes}
    year_by_scope = {s: ["scale(Year)"] for s in scopes}
    re_by_scope   = {s: ["(1|Country)"] for s in scopes}

st.subheader("Rscript configuration")
rscript_user = st.text_input("Path to Rscript.exe (optional)", value="")
rscript_exe  = find_rscript(rscript_user)

st.write({
    "Rscript_exe": rscript_exe,
    "exists": bool(rscript_exe and Path(rscript_exe).exists()),
})

if st.button("Build model grid"):
    grid = generate_model_grid(
        predictors=predictors,
        fixed_effects=fe_by_scope,
        scopes=scopes,
        dv="outbreak",
        max_predictors_per_model=6,
        min_predictors_per_model=1,
        year_terms_by_scope=year_by_scope,
        random_terms_by_scope=re_by_scope,
        engine=engine,
    )
    st.session_state["grid"] = grid
    st.success(f"Grid built with {len(grid):,} specs.")

if "grid" in st.session_state:
    st.dataframe(st.session_state["grid"].head(10), use_container_width=True)

if st.button("Write pending batches"):
    grid = st.session_state.get("grid")
    if grid is None:
        st.warning("Build the grid first.")
    else:
        pending = filter_pending_models(grid, output_dir=results_dir, merged_file=merged_out)
        st.write(f"Pending: {len(pending):,}")
        paths = write_grid_batches(pending, out_dir=specs_dir, basename="model_grid", batch_size=batch_size)
        st.success(f"Wrote {len(paths)} batch files in {specs_dir}.")

st.subheader("Run R on batches")
if st.button("Run R now"):
    app_path = Path(__file__).resolve()
    project_root = get_project_root(app_path.parent)
    r_script = project_root / "r" / "run_grid.R"

    specs_dir_path = Path(specs_dir).resolve()
    panel_abs   = str(Path(panel_path).resolve())
    results_abs = str(Path(results_dir).resolve())
    Path(results_abs).mkdir(parents=True, exist_ok=True)
    st.write(f"panel_abs: {panel_abs}")
    batch_files = sorted(p for p in specs_dir_path.glob("model_grid_*.csv"))

    pb = st.progress(0.0, text="Running R batches…")
    for i, b in enumerate(batch_files, start=1):
        b_abs = str(b.resolve())
        cmd = [
            rscript_exe,
            str(r_script),
            b_abs,
            panel_abs,
            results_abs
        ]
        st.code(" ".join(f'"{c}"' if " " in c else c for c in cmd), language="bash")

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                cwd=str(project_root),
                capture_output=True,
                text=True
            )
            if proc.stdout:
                st.text_area(f"R stdout ({b.name})", proc.stdout, height=120)
            if proc.stderr:
                st.text_area(f"R stderr ({b.name})", proc.stderr, height=120)
        except FileNotFoundError as e:
            st.error("Windows could not find a file from the above command (likely Rscript path).")
            st.exception(e)
            st.stop()
        except subprocess.CalledProcessError as e:
            st.error(f"R failed on {b.name} with exit code {e.returncode}. See stderr above.")
            if e.stdout:
                st.text_area(f"R stdout ({b.name})", e.stdout, height=120)
            if e.stderr:
                st.text_area(f"R stderr ({b.name})", e.stderr, height=200)
            st.stop()

        pb.progress(i / len(batch_files), text=f"Completed {i}/{len(batch_files)}")
    pb.empty()
    st.success("Finished R runs.")

# Merge results
if st.button("Merge results"):
    merged = merge_model_results(
        input_dir=results_dir,
        output_file=merged_out,
        pattern="batch_*.csv",
        delete_after=False
    )
    st.write(f"Merged rows: {len(merged):,}")
    st.dataframe(rank_models(merged).head(30), use_container_width=True)
    try:
        with open(merged_out, "rb") as fh:
            st.download_button(
                "Download merged results (as saved on disk)",
                data=fh.read(),
                file_name=os.path.basename(merged_out)
            )
    except FileNotFoundError:
        st.warning("Merged output file not found on disk yet.")
