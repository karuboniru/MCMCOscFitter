#include "BargerPropagator.h"
#include "StateI.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include "walker.h"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TF3.h>
#include <TMath.h>
#include <TRandom.h>
#include <cmath>
#include <functional>

// some default input to Prob3...
namespace {
double dcp_in = 0.;
double h_in = 1.0;
// int v_in = 1.0;
bool kSquared = true; // are we using sin^2(x) variables?

// int kNuBar = 1 * v_in;
double DM2 = h_in * 2.5e-3;
double Theta23 = 0.5;
double Theta13 = 0.0215;
double dm2 = 7.6e-5;
double Theta12 = 0.302;
// double delta   = 270.0 * (3.1415926/180.0);
double delta = dcp_in * (3.1415926 / 180.0);
} // namespace

namespace {
constexpr double Emin = 3, Emax = 10;
}

class NeutrinoState : virtual public StateI {
public:
  ~NeutrinoState() = default;

  NeutrinoState(std::function<double(double, double, double, int)> m_w)
      : weight_calculator(m_w) {};

  NeutrinoState(const NeutrinoState &) = default;
  NeutrinoState(NeutrinoState &&) = default;
  NeutrinoState &operator=(const NeutrinoState &) = default;
  NeutrinoState &operator=(NeutrinoState &&) = default;

  // default constructor, should not be used
  NeutrinoState() = default;

  double E, costheta, phi;
  int flavor;
  double weight;

  std::function<double(double, double, double, int)> weight_calculator;

  virtual void proposeStep() override {
    // E = TMath::Power(10, gRandom->Uniform(-1, 2));
    E = gRandom->Uniform(Emin, Emax);
    costheta = gRandom->Uniform(-1, 1);
    phi = gRandom->Uniform(0, 2 * TMath::Pi());
    switch (gRandom->Integer(4)) {
    case 0:
      flavor = -14;
      break;
    case 1:
      flavor = 14;
      break;
    case 2:
      flavor = 12;
      break;
    case 3:
      flavor = -12;
      break;
    }

    weight = weight_calculator(E, costheta, phi, flavor);
  }
  virtual double GetLogLikelihood() const override { return log(weight); }
};

class TimeCount {
public:
  TimeCount(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) {}
  ~TimeCount() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " musec\n";
  }

private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

int main() {
  // load flux
  HondaFlux honda{"/var/home/yan/neutrino/honda-3d.txt"};
  // honda.LoadFluxFile();
  // load cross section
  genie_xsec spline(
      "/var/home/yan/neutrino/spline/full/genie_spline_3_0_2/G18_02b_02_11b/"
      "3.04.02-routine_validation_01-xsec_vA/total_xsec.root");
  // spline.LoadSplineFile(
  //     );

  // // load BargerPropagator
  // BargerPropagator b;
  auto weight = [&](double E, double costheta, double phi, int flavor) {
    int symbol = flavor > 0 ? 1 : -1;
    double flux_numu = honda.GetFlux(E, costheta, phi, symbol * 14);
    double flux_nue = honda.GetFlux(E, costheta, phi, symbol * 12);
    double xsec = spline.GetXsec(E, flavor, 1000060120);

    BargerPropagator b;
    b.SetMNS(Theta12, Theta13, Theta23, dm2, DM2, delta, E, kSquared, flavor);
    b.DefinePath(costheta, 25);
    b.propagate(flavor);
    int prob3_flavor = symbol * (abs(flavor) == 14 ? 1 : 2);
    return (flux_numu * b.GetProb(symbol * 2, prob3_flavor) +
            flux_nue * b.GetProb(symbol * 1, prob3_flavor)) *
           xsec;
  };

  NeutrinoState init_state{weight};
  size_t e_count{};
  {
    TimeCount tc("weight_int");
    TF3 weight_func(
        "weight_func",
        [&weight](double *x, double *) {
          double E = x[0];
          double costh = x[1];
          double phi = x[2];
          return weight(E, costh, phi, 14) + weight(E, costh, phi, -14) +
                 weight(E, costh, phi, -12) + weight(E, costh, phi, 12);
        },
        Emin, Emax, -1, 1, 0, 2 * TMath::Pi(), 0);
    // weight_func.SetNpx(500);
    // weight_func.SetNpy(50);
    // weight_func.SetNpz(30);
    double weight_int =
        weight_func.Integral(Emin, Emax, -1, 1, 0, 2 * TMath::Pi(), 0.01);
    e_count = weight_int / 100;
    std::cout << "weight_int: " << weight_int << std::endl;
  }

  init_state.proposeStep();

  auto df =
      ROOT::RDataFrame{e_count}
          .Define("state",
                  [&]() {
                    for (size_t i{}; i < 5000; i++) {
                      NeutrinoState newState = init_state;
                      newState.proposeStep();
                      if (MCMCAcceptState(init_state, newState)) {
                        init_state = newState;
                      }
                    }
                    return init_state;
                  })
          .Define("E", [](const NeutrinoState &s) { return s.E; }, {"state"})
          .Define("costheta", [](const NeutrinoState &s) { return s.costheta; },
                  {"state"})
          .Define("phi", [](const NeutrinoState &s) { return s.phi; },
                  {"state"})
          .Define("w", [](const NeutrinoState &s) { return s.weight; },
                  {"state"})
          .Define("flavor", [](const NeutrinoState &s) { return s.flavor; },
                  {"state"})
      // .Snapshot("testevent", "testsample.root",
      //           {"E", "costheta", "phi", "flavor", "w"});
      ;
  ROOT::RDF::Experimental::AddProgressBar(df);
  df.Snapshot("testevent", "testsample.root",
              {"E", "costheta", "phi", "flavor", "w"});
}