import streamlit as st
from model_simulation_functions import simulate, extract_metrics, TNF_input
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Astrocyte-Neuron Interaction Simulator",
    layout="wide"
)

# Title
st.title("Astrocyte-Neuron Interaction Simulation")

# Sidebar controls
mode = st.sidebar.selectbox("TNF-α Profile Mode", options=["acute", "chronic"])
t_max = st.sidebar.slider("Simulation Time (s)", min_value=10, max_value=200, value=50, step=10)

# Model parameters input
st.sidebar.header("Model Parameters")
def_param = {
    'alpha': 0.1,
    'beta': 0.2,
    'eta': 0.05,
    'delta': 0.3,
    'epsilon': 0.1,
    'gamma': 0.4,
    'tau': 0.5,
    'tnf_amp': 1.0,
    'tnf_decay_fast': 0.5,
    'tnf_decay_slow': 5.0,
    'tnf_chronic_amp': 0.5,
    'tnf_chronic_rate': 0.2,
    'tnf_chronic_midpoint': 20,
}
params = {}
for key, default in def_param.items():
    params[key] = st.sidebar.number_input(label=key, value=default)

# Run simulation
if st.sidebar.button("Run Simulation"):
    t, Ca_astro, F_neuron = simulate(params, mode, t_max)
    metrics = extract_metrics(t, F_neuron)

    st.subheader("Simulation Metrics")
    st.write(metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, Ca_astro, label=r"Astrocyte Ca$^{2+}$")
    ax.plot(t, F_neuron, label=r"Neuron Firing Rate")
    tnf_vec = [TNF_input(tt, mode, params) for tt in t]
    ax.plot(t, tnf_vec, '--', label=rf"TNF-$\alpha$ ({mode.title()})", alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration / Firing Rate')
    ax.set_title(rf"{mode.title()} TNF-$\alpha$ Simulation")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
