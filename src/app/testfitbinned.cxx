#include "BinnedInteraction.h"
#include "MCMCWorker.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"
#include "timer.hpp"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TMath.h>
#include <TRandom.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <print>

int main(int argc, char **argv) {
  //   ROOT::EnableImplicitMT(10);
  TH1::AddDirectory(false);
  std::string outname = argc >= 2 ? argv[1] : "testfit.root";
  auto costheta_bins = linspace(-1., 1., FitConfig::n_costheta_bins + 1);
  auto Ebins = logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);

  BinnedInteraction bint{Ebins, costheta_bins, FitConfig::scale_factor,
                         FitConfig::E_rebin_factor, FitConfig::costh_rebin_factor,
                         FitConfig::ih_bias};
  auto cdata = bint.GenerateData();
  //   cdata.Round();

  using vals =
      std::tuple<double, double, double, double, double, double, size_t>;
  const auto nth = ROOT::GetThreadPoolSize() == 0 ? 1 : ROOT::GetThreadPoolSize();

  using MCMCChain = walker::MCMCWorker<BinnedInteraction>;
  std::vector<MCMCChain> chain_pool{};
  chain_pool.reserve(nth);
  for (size_t i = 0; i < nth; i++) {
    chain_pool.emplace_back(bint, cdata);
    chain_pool.back().step(cdata);  // warm-up step to diversify starting states
  }

  auto rawdf = ROOT::RDataFrame{135000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  std::atomic<size_t> count{};
  auto df =
      rawdf
          .Define("tuple",
                  [&chain_pool, &cdata, &count](unsigned int id) -> vals {
                    auto &chain = chain_pool[id];
                    TimeCount timer{"3 step"};
                    for (size_t i = 0; i < 3; i++) {
                      chain.step(cdata);
                    }
                    count++;
                    return std::make_tuple(
                        chain.state().GetDM32sq(), chain.state().GetDM21sq(),
                        chain.state().GetT12(),    chain.state().GetT13(),
                        chain.state().GetT23(),    chain.state().GetDeltaCP(),
                        count.load());
                  },
                  {"rdfslot_"})
          .Define("DM2", [](const vals &t) { return std::get<0>(t); },
                  {"tuple"})
          .Define("Dm2", [](const vals &t) { return std::get<1>(t); },
                  {"tuple"})
          .Define("T12", [](const vals &t) { return std::get<2>(t); },
                  {"tuple"})
          .Define("T13", [](const vals &t) { return std::get<3>(t); },
                  {"tuple"})
          .Define("T23", [](const vals &t) { return std::get<4>(t); },
                  {"tuple"})
          .Define("DCP", [](const vals &t) { return std::get<5>(t); },
                  {"tuple"})
          .Define("count", [](const vals &t) { return std::get<6>(t); },
                  {"tuple"});
  ;
  // xx
  df.Snapshot("tree", outname,
              {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "count"});

  //   cdata.SaveAs("data.root");

  return 0;
}