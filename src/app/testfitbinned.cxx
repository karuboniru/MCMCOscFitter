#include "BinnedInteraction.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "timer.hpp"
#include "walker.h"

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

int main(int argc, char **argv) {
  //   ROOT::EnableImplicitMT(10);
  TH1::AddDirectory(false);
  //   std::vector<double> Ebins;
  std::vector<double> costheta_bins{-1,  -.9, -.8, -.7, -.6, -.5, -.4,
                                    -.3, -.2, -.1, 0,   .1,  .2,  .3,
                                    .4,  .5,  .6,  .7,  .8,  .9,  1};
  //   double deltae = 1;
  //   for (double i = 0; i * deltae + 3 < 10; i += deltae) {
  //     Ebins.push_back(i * deltae + 3);
  //   }
  auto Ebins = logspace(1., 10., 300);
  BinnedInteraction bint{Ebins, costheta_bins, 0.01, 30};
  // bint.proposeStep();
  // bint.Print();
  auto cdata = bint.GenerateData();
  cdata.Round();
  std::cout << std::format("numu    count: {} \n", cdata.hist_numu.Integral())
            << std::format("numubar count: {} \n",
                           cdata.hist_numubar.Integral())
            << std::endl;

  using combined_type = ModelAndData<BinnedInteraction, SimpleDataHist>;
  using vals =
      std::tuple<double, double, double, double, double, double, size_t>;
  std::vector<combined_type> state_pool{};
  auto nth = ROOT::GetThreadPoolSize() == 0 ? 1 : ROOT::GetThreadPoolSize();
  for (size_t i = 0; i < nth; i++) {
    state_pool.emplace_back(bint, cdata).proposeStep();
  }
  auto rawdf = ROOT::RDataFrame{150000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  std::atomic<size_t> count{};
  auto df =
      rawdf
          .Define("tuple",
                  [&state_pool, &count](unsigned int id) -> vals {
                    auto &current_state = state_pool[id];
                    TimeCount timer{"5 step"};
                    for (size_t i = 0; i < 5; i++) {
                      auto new_state = current_state;
                      new_state.proposeStep();
                      if (MCMCAcceptState(current_state, new_state)) {
                        current_state = new_state;
                      }
                    }
                    count++;
                    return std::make_tuple(
                        current_state.GetModel().GetDM2(),
                        current_state.GetModel().GetDm2(),
                        current_state.GetModel().GetT12(),
                        current_state.GetModel().GetT13(),
                        current_state.GetModel().GetT23(),
                        current_state.GetModel().GetDeltaCP(), count.load());
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
  df.Snapshot("tree", "testfit2.root",
              {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "count"});

  cdata.SaveAs("data.root");

  return 0;
}