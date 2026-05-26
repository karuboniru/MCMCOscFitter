"""Bayesian hierarchy comparison using JAX + ArviZ.

Fits NH and IH hypotheses to NH Asimov data.  Two sampler choices:
  hmc       Custom HMCSampler (default) — fast, well-tuned
  numpyro   NumPyro NUTS — exact posterior, GPU-accelerated
  both      Run both and compare

Usage:
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_hierarchy_pymc.py --fine --samples 500 --warmup 200 --chains 4
"""

import sys, os, time, math, argparse
import numpy as np

if '--fp32' in sys.argv:
    os.environ['JAX_BARGER_FLOAT32'] = '1'; sys.argv.remove('--fp32')
if '--no-vram-workaround' in sys.argv:
    os.environ['JAX_BARGER_NO_VRAM_WORKAROUND'] = '1'; sys.argv.remove('--no-vram-workaround')

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

import mcmcoscfitter as mof
import jax, jax.numpy as jnp
import arviz as az
from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.barger import oscillation_probabilities
from jax_barger.event_rate import event_rate, rebin_2d
from jax_barger.pymc_model import build_jax_log_prob, fit_map, fit_numpyro_nuts_exact
from jax_barger.mcmc import HMCSampler, _PNAMES

_CHANNELS = ['numu','numubar','nue','nuebar']

parser = argparse.ArgumentParser(description='Hierarchy comparison with ArviZ')
parser.add_argument('--sampler', choices=['hmc','numpyro','both'], default='hmc')
parser.add_argument('--samples', type=int, default=500)
parser.add_argument('--warmup', type=int, default=200)
parser.add_argument('--chains', type=int, default=4)
parser.add_argument('--leapfrog', type=int, default=15)
parser.add_argument('--eps0', type=float, default=0.05)
parser.add_argument('--fast', action='store_true')
parser.add_argument('--fp32', action='store_true')
parser.add_argument('--no-vram-workaround', action='store_true')
args = parser.parse_args()

if args.fast:
    N_E,N_COS,ER,CR=10,12,1,1; ml="FAST (10x12)"
else:
    N_E,N_COS,ER,CR=200,120,20,10; ml=f"FINE ({N_E}x{N_COS} -> {ER}x{CR})"

EE=mof.logspace(0.1,20.0,N_E+1); CC=mof.linspace(-1.0,1.0,N_COS+1)
Ec=np.array(mof.to_center(EE)); Cc=np.array(mof.to_center(CC))
s=float(mof.scale_factor_6y); r,d,y=default_prem()
pi=mof.load_physics_input(EE,CC,s)
fl={k:jnp.array(pi[f'flux_{k}']) for k in _CHANNELS}
xs={k:jnp.array(pi[f'xsec_{k}']) for k in _CHANNELS}
dp,rp=precompute_path_data(jnp.array(Cc),r,d,y)

NH_T={'DM2':2.455e-3,'Dm2':7.53e-5,'T23':0.558,'T13':2.19e-2,'T12':0.307,'DCP':1.19*np.pi}
NH_S={'DM2':0.028e-3,'Dm2':0.18e-5,'T23':0.018,'T13':0.07e-2,'T12':0.013,'DCP':0.22*np.pi}
IH_T={'DM2':-2.529e-3,'Dm2':7.53e-5,'T23':0.553,'T13':2.19e-2,'T12':0.307,'DCP':1.19*np.pi}
IH_S={'DM2':0.029e-3,'Dm2':0.18e-5,'T23':0.020,'T13':0.07e-2,'T12':0.013,'DCP':0.22*np.pi}
def s2t(v): return np.arcsin(np.sqrt(np.clip(v,0,1)))
_nh=np.array([NH_T['DM2'],NH_T['Dm2'],s2t(NH_T['T23']),s2t(NH_T['T13']),NH_T['DCP'],s2t(NH_T['T12'])])
Pn=oscillation_probabilities(jnp.array(Ec),jnp.array(Cc),float(_nh[5]),float(_nh[3]),float(_nh[2]),float(_nh[4]),float(_nh[1]),float(_nh[0]),r,d,y)
ev=event_rate(np.array(Pn),fl,xs)
da={k:jnp.array(rebin_2d(ev[k],ER,CR)) for k in _CHANNELS}
MB=[(-1,1),(1e-7,1e-3),(0.1,np.pi/2-0.01),(0.01,np.pi/2-0.01),(-np.pi,np.pi),(0.1,np.pi/2-0.01)]

print(f"{'='*72}\nHierarchy Comparison -- JAX + ArviZ\n{'='*72}")
print(f"Grid: {ml}  Sampler: {args.sampler}  Chains: {args.chains}x{args.samples}")
for k in _CHANNELS: print(f"  {k:>8s} sum = {float(da[k].sum()):.1f}")


def prior_mass_diag(pm,ps,tm):
    th23=float(tm[2]);th13=float(tm[3]);th12=float(tm[5])
    d23=1./np.sin(2.*th23);d13=1./np.sin(2.*th13);d12=1./np.sin(2.*th12)
    pv=np.array([ps['DM2']**2,ps['Dm2']**2,(ps['T23']*d23)**2,(ps['T13']*d13)**2,ps['DCP']**2,(ps['T12']*d12)**2])
    return 1./np.maximum(pv,1e-30)


def do_fit(label,pm,ps):
    print(f"\n{'='*72}\n  {label} HYPOTHESIS\n{'='*72}")
    zp=np.array([pm.get(n,0) if n in ['DM2','Dm2','DCP'] else s2t(pm[n]) for n in _PNAMES])
    nlp=build_jax_log_prob(jnp.array(Ec),jnp.array(Cc),dp,rp,fl,xs,da,pm,ps,ER,CR)
    _=jax.jit(nlp)(jnp.array(zp)); _=jax.jit(jax.grad(nlp))(jnp.array(zp))
    print("  Finding MAP..."); t0=time.time()
    me,mr=fit_map(nlp,zp,bounds=MB,maxiter=500)
    print(f"  MAP NLLP={mr.fun:.4f}  evals={mr.nfev}  time={time.time()-t0:.1f}s")

    if args.sampler in ('hmc','both'):
        imd=prior_mass_diag(pm,ps,me)
        np.random.seed(hash(label)%2**31)
        pt=np.random.randn(6)*0.3
        zi=np.array(me)+pt*np.array([ps['DM2'],ps['Dm2'],0.05,0.01,0.1,0.05])
        zi[2]=np.clip(zi[2],.02,np.pi/2-.02);zi[3]=np.clip(zi[3],.002,np.pi/2-.002)
        zi[5]=np.clip(zi[5],.02,np.pi/2-.02);zi[4]=np.arctan2(np.sin(zi[4]),np.cos(zi[4]))
        sa=HMCSampler(nlp,eps_0=args.eps0,n_leapfrog=args.leapfrog,target_accept=0.651,initial_mass_diag=imd)
        print(f"  HMC warmup ({args.warmup})..."); t0=time.time()
        sa.warmup(n_steps=args.warmup,z_init=zi,adapt_step=True,adapt_mass=False)
        print(f"  time={time.time()-t0:.1f}s")
        print(f"  HMC sample ({args.samples}x{args.chains})..."); t0=time.time()
        ch=sa.sample(n_samples=args.samples,n_chains=args.chains)
        print(f"  time={time.time()-t0:.1f}s"); sa.diagnostics()
        post={name:np.array(ch[:,:,i]) for i,name in enumerate(_PNAMES)}
        idt=az.from_dict({"posterior":post},sample_dims=["chain","draw"])
    elif args.sampler=='numpyro':
        print(f"  NUTS warmup ({args.warmup})..."); t0=time.time()
        idt=fit_numpyro_nuts_exact(nlp,pm,ps,n_draws=args.samples,n_warmup=args.warmup,n_chains=args.chains,target_accept=0.9,seed=hash(label)%2**31)
        print(f"  time={time.time()-t0:.1f}s")

    print(f"  Posterior:")
    for i,name in enumerate(_PNAMES):
        m=float(idt.posterior[name].mean());s=float(idt.posterior[name].std())
        print(f"    {name}: {m:.4e}+/-{s:.1e}")
    return idt,me


res_nh,mu_nh=do_fit("NH",NH_T,NH_S)
res_ih,mu_ih=do_fit("IH",IH_T,IH_S)

print(f"\n{'='*72}\nMODEL COMPARISON\n{'='*72}")
print(f"  NH MAP NLLP = {float(jax.jit(build_jax_log_prob(jnp.array(Ec),jnp.array(Cc),dp,rp,fl,xs,da,NH_T,NH_S,ER,CR))(jnp.array(mu_nh))):.2f}")
print(f"  IH MAP NLLP = {float(jax.jit(build_jax_log_prob(jnp.array(Ec),jnp.array(Cc),dp,rp,fl,xs,da,IH_T,IH_S,ER,CR))(jnp.array(mu_ih))):.2f}")
for name in _PNAMES:
    nh_m=res_nh.posterior[name].mean().values; nh_s=res_nh.posterior[name].std().values
    ih_m=res_ih.posterior[name].mean().values; ih_s=res_ih.posterior[name].std().values
    print(f"  {name}: NH={nh_m:.4e}+/-{nh_s:.1e}  IH={ih_m:.4e}+/-{ih_s:.1e}")

for res,lab in [(res_nh,'nh'),(res_ih,'ih')]:
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),f'pymc_fit_{lab}.nc')
    res.to_netcdf(out); print(f"  Saved {lab} to {out}")

print(f"\n{'='*72}\nDone.\n{'='*72}")
