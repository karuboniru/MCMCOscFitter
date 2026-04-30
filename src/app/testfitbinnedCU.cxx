#include "MCMCWorker.h"
#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>

#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TRandom.h>
#include <cmath>
#include <print>

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  std::string outname = argc >= 2 ? argv[1] : "testfit.root";
  auto costheta_bins = linspace(-1., 1., FitConfig::n_costheta_bins + 1);
  auto Ebins = logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);

  ParBinnedInterface bint{Ebins, costheta_bins, FitConfig::scale_factor,
                           FitConfig::E_rebin_factor, FitConfig::costh_rebin_factor,
                           FitConfig::ih_bias};
  SimpleDataHist data = bint.GenerateData();

  walker::MCMCWorker<ParBinnedInterface> chain{bint, data};

  using vals =
      std::tuple<double, double, double, double, double, double, size_t>;

  auto rawdf = ROOT::RDataFrame{1250000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  size_t count = 0;

  auto df =
      rawdf
          .Define("tuple",
                  [&chain, &data, &count]() -> vals {
                    for (size_t i = 0; i < 5; i++) {
                      chain.step(data);
                    }
                    count++;
                    return std::make_tuple(
                        chain.state().GetDM32sq(), chain.state().GetDM21sq(),
                        chain.state().GetT12(),    chain.state().GetT13(),
                        chain.state().GetT23(),    chain.state().GetDeltaCP(),
                        count);
                  })
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

  df.Snapshot("tree", outname,
              {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "count"});

  return 0;
}
