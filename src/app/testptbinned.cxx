#include "BinnedInteraction.h"
#include "ParallelTempering.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"
#include "temperature_ladder.h"
#include "timer.hpp"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TMath.h>
#include <TRandom.h>
#include <cmath>
#include <memory>
#include <print>
#include <vector>

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  std::string outname = argc >= 2 ? argv[1] : "testpt.root";
  const size_t n_chains =
      (argc >= 3) ? std::stoul(argv[2])
                  : 8;  // number of temperature rungs
  const size_t n_iterations = (argc >= 4) ? std::stoul(argv[3]) : 10000;

  auto costheta_bins =
      linspace(-1., 1., FitConfig::n_costheta_bins + 1);
  auto Ebins =
      logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);

  BinnedInteraction bint{Ebins, costheta_bins, FitConfig::scale_factor,
                         FitConfig::E_rebin_factor,
                         FitConfig::costh_rebin_factor, FitConfig::ih_bias};
  auto cdata = bint.GenerateData();

  // Build temperature ladder: geometric spacing from 1.0 to 200.0.
  auto ladder = mcmc::TemperatureLadder::geometric(n_chains, 1.0, 200.0);
  std::print("Temperature ladder ({} chains):\n", ladder.size());
  for (size_t i = 0; i < ladder.size(); ++i)
    std::print("  chain {:2d}: T = {:8.2f}, beta = {:.4f}\n", i, ladder[i],
               ladder.beta(i));

  using PT = walker::ParallelTempering<BinnedInteraction>;
  PT pt(ladder, bint, cdata);
  pt.set_swap_interval(100);

  // Warmup: one step per chain.
  for (size_t i = 0; i < pt.num_chains(); ++i)
    pt.chain(i).step(cdata);

  // Prepare output.
  std::vector<double> dm32, dm21, t12, t13, t23, dcp;
  std::vector<size_t> step_nos;
  dm32.reserve(n_iterations);
  dm21.reserve(n_iterations);
  t12.reserve(n_iterations);
  t13.reserve(n_iterations);
  t23.reserve(n_iterations);
  dcp.reserve(n_iterations);
  step_nos.reserve(n_iterations);

  TimeCount timer{"PT run"};

  pt.run(cdata, n_iterations,
         [&](const BinnedInteraction &state, size_t step) {
           dm32.push_back(state.GetDM32sq());
           dm21.push_back(state.GetDM21sq());
           t12.push_back(state.GetT12());
           t13.push_back(state.GetT13());
           t23.push_back(state.GetT23());
           dcp.push_back(state.GetDeltaCP());
           step_nos.push_back(step);
         });

  // timer prints on destruction (RAII)

  // Print swap acceptance rates.
  std::print("\nSwap acceptance rates (interval = {}):\n", pt.swap_interval());
  for (size_t i = 0; i < pt.num_chains() - 1; ++i) {
    std::print("  ({}<->{}): T={:.1f}<->T={:.1f}  rate = {:.3f}\n", i, i + 1,
               ladder[i], ladder[i + 1], pt.swap_acceptance_rate(i));
  }

  // Write tree.
  ROOT::RDataFrame df(static_cast<int>(dm32.size()));
  auto result = df.Define("DM2", [&dm32](ULong64_t e) { return dm32[e]; })
                    .Define("Dm2",
                            [&dm21](ULong64_t e) { return dm21[e]; })
                    .Define("T12",
                            [&t12](ULong64_t e) { return t12[e]; })
                    .Define("T13",
                            [&t13](ULong64_t e) { return t13[e]; })
                    .Define("T23",
                            [&t23](ULong64_t e) { return t23[e]; })
                    .Define("DCP",
                            [&dcp](ULong64_t e) { return dcp[e]; })
                    .Define("step", [&step_nos](ULong64_t e) {
                      return step_nos[e];
                    });
  result.Snapshot("tree", outname,
                  {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "step"});

  std::print("Results written to {}\n", outname);
  return 0;
}
