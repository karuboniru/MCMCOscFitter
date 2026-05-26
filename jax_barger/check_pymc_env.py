"""Verify PyMC + JAX environment for the jax_barger integration."""

import sys, os

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

print("=" * 60)
print("PyMC + JAX Environment Check")
print("=" * 60)

# 1 -- JAX devices
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if hasattr(d, 'platform') and d.platform == 'gpu']
cpu_devices = [d for d in devices if hasattr(d, 'platform') and d.platform == 'cpu']
print(f"\nJAX devices: {len(devices)} total ({len(gpu_devices)} GPU, {len(cpu_devices)} CPU)")
for d in devices:
    print(f"  {d}")

# 2 -- PyMC imports
print("\n--- PyMC imports ---")
import pymc as pm
print(f"  PyMC version:  {pm.__version__}")

import numpyro
print(f"  NumPyro version: {numpyro.__version__}")

try:
    import nutpie
    print(f"  nutpie version: {nutpie.__version__}")
except ImportError:
    print("  nutpie: NOT AVAILABLE")

import arviz as az
print(f"  ArviZ version: {az.__version__}")

# 3 -- pytensor.link.jax.dispatch
print("\n--- pytensor.link.jax.dispatch ---")
try:
    from pytensor.link.jax.dispatch import jax_funcify
    print("  jax_funcify: OK")
except ImportError as e:
    print(f"  jax_funcify: FAILED ({e})")
    sys.exit(1)

# 4 -- Minimal JAX Op registration smoke test
print("\n--- Custom JAX Op smoke test ---")
import pytensor.tensor as pt
from pytensor.graph import Op, Apply
import numpy as np

class _SumSqOp(Op):
    __props__ = ()
    def make_node(self, x, y):
        return Apply(self, [pt.as_tensor_variable(x), pt.as_tensor_variable(y)], [pt.scalar()])
    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.array(float(inputs[0])**2 + float(inputs[1])**2, dtype='float64')

def _sumsq_jax_impl(op, **kwargs):
    def f(x, y):
        return x**2 + y**2
    return f

jax_funcify.register(_SumSqOp, _sumsq_jax_impl)

op = _SumSqOp()
x_sym, y_sym = pt.scalars('x', 'y')
f_sym = op(x_sym, y_sym)
from pytensor.compile.function import function as pt_function
f_fn = pt_function([x_sym, y_sym], f_sym)
result = f_fn(3.0, 4.0)
print(f"  Custom Op f(3,4) = {result:.1f} (expected 25.0)")
assert abs(result - 25.0) < 0.5, f"FAILED: got {result}"
print("  Custom JAX Op: OK")

# 5 -- NUTS with numpyro smoke test (2D Gaussian)

# 5 -- NUTS with numpyro smoke test (2D Gaussian)
print("\n--- NUTS+numpyro smoke test ---")
print("\n--- NUTS+numpyro smoke test ---")
import jax.numpy as jnp

with pm.Model() as gauss2d:
    mu = pm.Normal("mu", mu=0.0, sigma=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=np.array([1.0, 1.5, 0.8, 1.2, 1.1]))

with gauss2d:
    trace = pm.sample(draws=200, tune=200, chains=2,
                      nuts_sampler="numpyro",
                      random_seed=42,
                      progressbar=False)

print(f"  Posterior mu mean:    {float(trace.posterior['mu'].mean()):.4f}")
print(f"  Posterior sigma mean: {float(trace.posterior['sigma'].mean()):.4f}")
print("  NUTS+numpyro: OK")

print(f"\n{'=' * 60}")
print("All checks passed! Environment is ready.")
print("=" * 60)
