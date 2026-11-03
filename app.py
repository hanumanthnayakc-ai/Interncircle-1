"""
Interactive demo: move the sliders, the decision tree re-predicts instantly.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from predictor import Predictor

st.set_page_config(page_title="Tumour classifier demo", layout="wide")


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


P = get_predictor()
M = P.meta["metrics"]


def key(feature: str) -> str:
    return f"slider_{feature}"


# Initialise slider state once (before the widgets are created).
for f, v in P.defaults().items():
    st.session_state.setdefault(key(f), v)


def load_preset(kind: str) -> None:
    for f, v in P.preset(kind).items():
        st.session_state[key(f)] = v


def reset() -> None:
    for f, v in P.defaults().items():
        st.session_state[key(f)] = v


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Try an example")
    st.button("Typical benign tumour", on_click=load_preset, args=("benign",))
    st.button("Typical malignant tumour", on_click=load_preset, args=("malignant",))
    st.button("Reset to dataset medians", on_click=reset)
    st.divider()
    st.header("About the model")
    st.write(f"Decision tree, depth {P.meta['depth']}, trained on the Breast Cancer Wisconsin "
             f"dataset using {len(P.features)} measurements.")
    st.write(f"On {M['test_rows']} unseen test cases: accuracy {M['accuracy']:.1%}, "
             f"caught {M['malignant_recall']:.1%} of malignant tumours.")
    st.caption("Educational demo only. Not a medical device and not medical advice.")

# ---- main page ---------------------------------------------------------------
st.title("Tumour classifier: live prediction")
st.write("Adjust the cell measurements and watch the prediction update.")

left, right = st.columns([1, 1], gap="large")

values = {}
with left:
    st.subheader("Measurements")
    for f in P.features:
        spec = P.slider_spec(f)
        values[f] = st.slider(f.capitalize(), min_value=spec["min"], max_value=spec["max"],
                              step=spec["step"], format=spec["format"], key=key(f))

result = P.predict(values)

with right:
    st.subheader("Prediction")
    if result["label"] == "malignant":
        st.error("Predicted: MALIGNANT")
    else:
        st.success("Predicted: BENIGN")
    st.metric("Probability malignant", f"{result['p_malignant']:.0%}")
    st.progress(result["p_malignant"])
    st.caption("A decision tree's probability is the share of similar training cases in the "
               "final leaf, so it moves in steps rather than smoothly. The tree also judges "
               "combinations of measurements, so a single extreme slider may not flip the "
               "prediction on its own; the explanation below shows which questions decided it.")

    with st.expander("Why this prediction?", expanded=True):
        for r in result["rules"]:
            sign = "<=" if r["went_left"] else ">"
            st.write(f"- **{r['feature']}** = {r['value']:.4g} is {sign} {r['threshold']:.4g}")

    with st.expander("How do my values compare?"):
        table = pd.DataFrame(P.comparison(values)).set_index("Measurement")
        st.table(table)
