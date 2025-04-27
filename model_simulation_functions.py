import numpy as np
from scipy.integrate import odeint, simpson
from functools import lru_cache
from typing import Dict, Tuple, Callable, List

# Type aliases
ParamsType = Dict[str, float]
StateType = List[float]
TimeVector = np.ndarray
ConcentrationVector = np.ndarray
TNFFunction = Callable[[float], float]

def _make_hashable(params: ParamsType) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted(params.items()))

@lru_cache(maxsize=10)
def _cached_simulate(params_tuple: Tuple[Tuple[str, float], ...], mode: str='acute', t_max: int=50):
    params = dict(params_tuple)
    t = np.linspace(0, t_max, 500)
    tnf_profile = lambda time_point: TNF_input(time_point, mode, params)
    y0 = [0.0, 0.0]
    sol = odeint(model_odes, y0, t, args=(params, tnf_profile))
    return t, sol[:,0], sol[:,1]

def simulate(params: ParamsType, mode: str='acute', t_max: int=50):
    return _cached_simulate(_make_hashable(params), mode, t_max)

def TNF_input(t: float, mode: str='acute', params: ParamsType=None) -> float:
    if params is None:
        raise ValueError("Parameters required for TNF_input.")
    if mode=='acute':
        tnf_amp = params['tnf_amp']
        fast = params['tnf_decay_fast']
        slow = params['tnf_decay_slow']
        if fast<=0 or slow<=0 or fast>=slow:
            return 0.0
        return tnf_amp*(np.exp(-t/slow)-np.exp(-t/fast))
    elif mode=='chronic':
        amp=params['tnf_chronic_amp']
        rate=params['tnf_chronic_rate']
        mid=params['tnf_chronic_midpoint']
        expn=-rate*(t-mid)
        if expn>700: return 0.0
        return amp/(1+np.exp(expn))
    else:
        raise ValueError("Mode must be 'acute' or 'chronic'.")

def model_odes(y: StateType, t: float, params: ParamsType, tnf_profile: TNFFunction):
    Ca, F = y
    TNF_t = tnf_profile(t)
    dCa = params['alpha']*TNF_t + params['eta']*F - params['beta']*Ca
    Glu = params['gamma']*Ca
    tau=params['tau']
    if tau<=0: raise ValueError("tau must be positive.")
    dF=(params['delta']*Glu - params['epsilon'] - F)/tau
    return [dCa, dF]

def extract_metrics(t, F_neuron):
    if len(t)!=len(F_neuron) or len(t)<2:
        raise ValueError("Time and firing vectors mismatch.")
    peak=F_neuron.max()
    auc=simpson(F_neuron, x=t)
    ti_peak=t[F_neuron.argmax()]
    dt=t[1]-t[0]
    duration=(F_neuron>1.0).sum()*dt
    return {'peak_firing':peak,'auc_firing':auc,'time_to_peak':ti_peak,'firing_duration':duration}
