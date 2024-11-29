#include "ParProb3ppOscillation.h"
#ifndef __CUDA__
#include <cpupropagator.hpp> // include openmp propagator
#else
#include <cudapropagator.cuh> // include cuda propagator
#endif

#include <thread>

#include <array>
#include <memory>

const size_t n_threads_propagator = std::thread::hardware_concurrency();

ParProb3ppOscillation::ParProb3ppOscillation(const std::vector<float> &Ebin,
                                             const std::vector<float> &costhbin)
    :
#ifndef __CUDA__
      propagator_neutrino{std::make_unique<cudaprob3::CpuPropagator<float>>(
          (int)costhbin.size(), (int)Ebin.size(), n_threads_propagator)},
      propagator_antineutrino{std::make_unique<cudaprob3::CpuPropagator<float>>(
          (int)costhbin.size(), (int)Ebin.size(), n_threads_propagator)}
#else
      propagator_neutrino{
          std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
              0, (int)costhbin.size(), (int)Ebin.size())},
      propagator_antineutrino{
          std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
              0, (int)costhbin.size(), (int)Ebin.size())}
#endif
      ,
      Ebins(Ebin), costheta_bins(costhbin) {
  load_state(*propagator_neutrino);
  load_state(*propagator_antineutrino);

  propagator_neutrino->calculateProbabilities(cudaprob3::Neutrino);
  propagator_antineutrino->calculateProbabilities(cudaprob3::Antineutrino);
}

ParProb3ppOscillation::ParProb3ppOscillation(const ParProb3ppOscillation &from)
    : OscillationParameters(from),
#ifndef __CUDA__
      propagator_neutrino{std::make_unique<cudaprob3::CpuPropagator<float>>(
          *from.propagator_neutrino)},
      propagator_antineutrino{std::make_unique<cudaprob3::CpuPropagator<float>>(
          *from.propagator_antineutrino)},
#else
      propagator_neutrino{
          std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
              0, (int)from.costheta_bins.size(), (int)from.Ebins.size(),
              from.costh_rebin_fac)},
      propagator_antineutrino{
          std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
              0, (int)from.costheta_bins.size(), (int)from.Ebins.size(),
              from.costh_rebin_fac)},
#endif
      Ebins(from.Ebins), costheta_bins(from.costheta_bins) {
#ifdef __CUDA__
  load_state(*propagator_neutrino);
  load_state(*propagator_antineutrino);
#endif
}

ParProb3ppOscillation &
ParProb3ppOscillation::operator=(const ParProb3ppOscillation &from) {
  OscillationParameters::operator=(from);
  Ebins = from.Ebins;
  costheta_bins = from.costheta_bins;
#ifndef __CUDA__
  *propagator_neutrino = *from.propagator_neutrino;
  *propagator_antineutrino = *from.propagator_antineutrino;
#else
  propagator_neutrino =
      std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
          0, (int)costheta_bins.size(), (int)Ebins.size(), costh_rebin_fac);
  propagator_antineutrino =
      std::make_unique<cudaprob3::CudaPropagatorSingle<float>>(
          0, (int)costheta_bins.size(), (int)Ebins.size(), costh_rebin_fac);
  load_state(*propagator_neutrino);
  load_state(*propagator_antineutrino);
#endif
  return *this;
}

void ParProb3ppOscillation::proposeStep() {
  OscillationParameters::proposeStep();
  re_calculate();
}

void ParProb3ppOscillation::re_calculate() {
  propagator_neutrino->setMNSMatrix(asin(sqrt(GetT12())), asin(sqrt(GetT13())),
                                    asin(sqrt(GetT23())), GetDeltaCP());
  propagator_neutrino->setNeutrinoMasses(GetDm2(), GetDM2());

  propagator_antineutrino->setMNSMatrix(asin(sqrt(GetT12())),
                                        asin(sqrt(GetT13())),
                                        asin(sqrt(GetT23())), GetDeltaCP());
  propagator_antineutrino->setNeutrinoMasses(GetDm2(), GetDM2());

  propagator_neutrino->calculateProbabilities(cudaprob3::Neutrino);
  propagator_antineutrino->calculateProbabilities(cudaprob3::Antineutrino);
}

std::array<std::array<double, 3>, 3>
ParProb3ppOscillation::GetProb(int flavor, double E, double costheta) const {
  throw std::runtime_error("Not implemented");
}

///> The 3D probability histogram
///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-mu][to: 0-e, 1-mu]
[[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
ParProb3ppOscillation::GetProb_Hist(std::vector<double> Ebin,
                                    std::vector<double> costhbin) {
  std::array<std::array<std::array<TH2D, 2>, 2>, 2> ret{};
  constexpr auto type_matrix = std::to_array(
      {std::to_array({cudaprob3::ProbType::e_e, cudaprob3::ProbType::e_m}),
       std::to_array({cudaprob3::ProbType::m_e, cudaprob3::ProbType::m_m})});

  for (int i = 0; i < 2; ++i) {
    auto &this_propagator =
        i == 0 ? propagator_neutrino : propagator_antineutrino;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        auto &this_prob = ret[i][j][k];
        auto &this_term = type_matrix[j][k];
        this_prob = TH2D("", "", Ebin.size() - 1, Ebin.data(),
                         costhbin.size() - 1, costhbin.data());
        // auto prob = this_propagator.getProbability(j,k, this_term);

#pragma omp parallel for
        for (size_t energy_bin_index = 0; energy_bin_index < Ebin.size() - 1;
             ++energy_bin_index) {
          for (size_t out_hist_costh_index{};
               out_hist_costh_index < costhbin.size() - 1;
               ++out_hist_costh_index) {
            this_prob.SetBinContent(
                energy_bin_index + 1, out_hist_costh_index + 1,
                this_propagator->getProbability(out_hist_costh_index,
                                                energy_bin_index, this_term));
          }
        }
      }
    }
  }
  return ret;
}

ParProb3ppOscillation::~ParProb3ppOscillation() = default;

#ifndef __CUDA__
void ParProb3ppOscillation::load_state(cudaprob3::CpuPropagator<float> &to_load)
#else
void ParProb3ppOscillation::load_state(
    cudaprob3::CudaPropagatorSingle<float> &to_load)
#endif
{
  to_load.setCosineList(costheta_bins);
  to_load.setEnergyList(Ebins);
  to_load.setDensity({0, 1220, 3480, 5701, 6371}, {13, 13, 11.3, 5, 3.3});
  // to_load.setDensityFromFile("/var/home/yan/code/MCMCOscFitter/external/"
  //                            "CUDAProb3/example/models/PREM_10layer.dat");
  to_load.setProductionHeight(15.0);
  to_load.setMNSMatrix(asin(sqrt(GetT12())), asin(sqrt(GetT13())),
                       asin(sqrt(GetT23())), GetDeltaCP());
  to_load.setNeutrinoMasses(GetDm2(), GetDM2());
}
