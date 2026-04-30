#include "BinnedInteraction.h"
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

  std::vector<BinnedInteraction> model_pool{};
  std::vector<double>            llh_cache{};
  model_pool.reserve(nth);
  llh_cache.reserve(nth);
  for (size_t i = 0; i < nth; i++) {
    model_pool.emplace_back(bint);
    model_pool.back().proposeStep();
    llh_cache.push_back(model_pool.back().GetLogLikelihood()
                       + model_pool.back().GetLogLikelihoodAgainstData(cdata));
  }

  auto rawdf = ROOT::RDataFrame{135000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  std::atomic<size_t> count{};
  auto df =
      rawdf
          .Define("tuple",
                  [&model_pool, &llh_cache, &cdata, &count](unsigned int id) -> vals {
                    auto &current = model_pool[id];
                    double &cur_llh = llh_cache[id];
                    TimeCount timer{"3 step"};
                    for (size_t i = 0; i < 3; i++) {
                      auto proposed = current;
                      proposed.proposeStep();

                      double nxt_llh = proposed.GetLogLikelihood()
                                     + proposed.GetLogLikelihoodAgainstData(cdata);

                      if (nxt_llh > cur_llh ||
                          gRandom->Rndm() < std::exp(nxt_llh - cur_llh)) {
                        current = std::move(proposed);
                        cur_llh = nxt_llh;
                      }
                    }
                    count++;
                    return std::make_tuple(
                        current.GetDM32sq(), current.GetDM21sq(),
                        current.GetT12(),    current.GetT13(),
                        current.GetT23(),    current.GetDeltaCP(),
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