#include "ParProb3ppOscillation.h"
#include <print>
#ifndef __CUDA__
#include <cpupropagator.hpp> // include openmp propagator
#else
#include <cudapropagator.cuh> // include cuda propagator
#endif

#include <thread>

#include "constants.h"
#include <array>
#include <memory>

const size_t n_threads_propagator = std::thread::hardware_concurrency() / 2;

ParProb3ppOscillation::ParProb3ppOscillation(
    const std::vector<oscillaton_calc_precision> &Ebin,
    const std::vector<oscillaton_calc_precision> &costhbin)
    :
#ifndef __CUDA__
      propagator_neutrino{
          std::make_unique<cudaprob3::CpuPropagator<oscillaton_calc_precision>>(
              (int)costhbin.size(), (int)Ebin.size(), n_threads_propagator)},
      propagator_antineutrino{
          std::make_unique<cudaprob3::CpuPropagator<oscillaton_calc_precision>>(
              (int)costhbin.size(), (int)Ebin.size(), n_threads_propagator)}
#else
      propagator_neutrino{std::make_unique<
          cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
          0, (int)costhbin.size(), (int)Ebin.size())},
      propagator_antineutrino{std::make_unique<
          cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
          0, (int)costhbin.size(), (int)Ebin.size())}
#endif
      ,
      Ebins(Ebin), costheta_bins(costhbin) {
  load_state(*propagator_neutrino, true);
  propagator_neutrino->calculateProbabilities(cudaprob3::Neutrino);

  load_state(*propagator_antineutrino, true);
  propagator_antineutrino->calculateProbabilities(cudaprob3::Antineutrino);
}

ParProb3ppOscillation::ParProb3ppOscillation(const ParProb3ppOscillation &from)
    : OscillationParameters(from),
#ifndef __CUDA__
      propagator_neutrino{
          std::make_unique<cudaprob3::CpuPropagator<oscillaton_calc_precision>>(
              *from.propagator_neutrino)},
      propagator_antineutrino{
          std::make_unique<cudaprob3::CpuPropagator<oscillaton_calc_precision>>(
              *from.propagator_antineutrino)},
#else
      propagator_neutrino{std::make_unique<
          cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
          0, (int)from.costheta_bins.size(), (int)from.Ebins.size(),
          from.costh_rebin_fac)},
      propagator_antineutrino{std::make_unique<
          cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
          0, (int)from.costheta_bins.size(), (int)from.Ebins.size(),
          from.costh_rebin_fac)},
#endif
      Ebins(from.Ebins), costheta_bins(from.costheta_bins) {
#ifdef __CUDA__
  load_state(*propagator_neutrino, false);
  load_state(*propagator_antineutrino, false);
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
  propagator_neutrino = std::make_unique<
      cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
      0, (int)costheta_bins.size(), (int)Ebins.size(), costh_rebin_fac);
  propagator_antineutrino = std::make_unique<
      cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>(
      0, (int)costheta_bins.size(), (int)Ebins.size(), costh_rebin_fac);
  load_state(*propagator_neutrino, true);
  load_state(*propagator_antineutrino, true);
#endif
  return *this;
}

void ParProb3ppOscillation::proposeStep() {
  OscillationParameters::proposeStep();
  re_calculate();
}

void ParProb3ppOscillation::re_calculate() {
  load_state(*propagator_neutrino, false);
  propagator_neutrino->calculateProbabilities(cudaprob3::Neutrino);

  load_state(*propagator_antineutrino, false);
  propagator_antineutrino->calculateProbabilities(cudaprob3::Antineutrino);
}

std::array<std::array<double, 3>, 3>
ParProb3ppOscillation::GetProb(int flavor, double E, double costheta) const {
  throw std::runtime_error("Not implemented");
}

///> The 3D probability histogram
///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-mu][to: 0-e, 1-mu]
[[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
ParProb3ppOscillation::GetProb_Hists(const std::vector<double> &Ebin,
                                     const std::vector<double> &costhbin) {
  std::array<std::array<std::array<TH2D, 2>, 2>, 2> ret{};
  constexpr auto type_matrix = std::to_array(
      {std::to_array({cudaprob3::ProbType::e_e, cudaprob3::ProbType::e_m}),
       std::to_array({cudaprob3::ProbType::m_e, cudaprob3::ProbType::m_m})});

  for (int i = 0; i < 2; ++i) {
    auto &this_propagator =
        i == 0 ? propagator_neutrino : propagator_antineutrino;
    // auto &this_type_matrix = i == 0 ? type_matrix : type_matrix_invert;
    auto &this_type_matrix = type_matrix;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        auto &this_prob = ret[i][j][k];
        auto &this_term = this_type_matrix[j][k];
        this_prob = TH2D("", "", Ebin.size() - 1, Ebin.data(),
                         costhbin.size() - 1, costhbin.data());
#pragma omp parallel for collapse(2)
        for (size_t energy_bin_index = 0; energy_bin_index < Ebin.size() - 1;
             ++energy_bin_index) {
          for (size_t out_hist_costh_index = 0;
               out_hist_costh_index < costhbin.size() - 1;
               ++out_hist_costh_index) {
            this_prob.SetBinContent(
                energy_bin_index + 1, out_hist_costh_index + 1,
                this_propagator->getProbability(out_hist_costh_index,
                                                energy_bin_index, this_term));
            // auto prob_sum =
            //     this_propagator->getProbability(out_hist_costh_index,
            //                                     energy_bin_index,
            //                                     cudaprob3::ProbType::e_e) +
            //     this_propagator->getProbability(out_hist_costh_index,
            //                                     energy_bin_index,
            //                                     cudaprob3::ProbType::e_m) +
            //     this_propagator->getProbability(out_hist_costh_index,
            //                                     energy_bin_index,
            //                                     cudaprob3::ProbType::e_t);
            // if (abs(prob_sum - 1) > 1e-7) {
            //   std::println(
            //       std::cerr,
            //       "Sum of probabilities is not 1: {:.5f} at E = {}, costh = "
            //       "{}",
            //       prob_sum, Ebin[energy_bin_index],
            //       costhbin[out_hist_costh_index]);
            // }
          }
        }
      }
    }
  }
  return ret;
}

///> The 3D probability histogram
///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-mu, 2-tau][to: 0-e, 1-mu,
/// 2-tau]
[[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
ParProb3ppOscillation::GetProb_Hists_3F(const std::vector<double> &Ebin,
                                        const std::vector<double> &costhbin) {
  std::array<std::array<std::array<TH2D, 3>, 3>, 2> ret{};
  constexpr auto type_matrix = std::to_array(
      {std::to_array({cudaprob3::ProbType::e_e, cudaprob3::ProbType::e_m,
                      cudaprob3::ProbType::e_t}),
       std::to_array({cudaprob3::ProbType::m_e, cudaprob3::ProbType::m_m,
                      cudaprob3::ProbType::m_t}),
       std::to_array({cudaprob3::ProbType::t_e, cudaprob3::ProbType::t_m,
                      cudaprob3::ProbType::t_t})});

  for (int i = 0; i < 2; ++i) {
    auto &this_propagator =
        i == 0 ? propagator_neutrino : propagator_antineutrino;
    // auto &this_type_matrix = i == 0 ? type_matrix : type_matrix_invert;
    auto &this_type_matrix = type_matrix;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        auto &this_prob = ret[i][j][k];
        auto &this_term = this_type_matrix[j][k];
        this_prob = TH2D("", "", Ebin.size() - 1, Ebin.data(),
                         costhbin.size() - 1, costhbin.data());
#pragma omp parallel for collapse(2)
        for (size_t energy_bin_index = 0; energy_bin_index < Ebin.size() - 1;
             ++energy_bin_index) {
          for (size_t out_hist_costh_index = 0;
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
void ParProb3ppOscillation::load_state(
    cudaprob3::CpuPropagator<oscillaton_calc_precision> &to_load, bool init)
#else
void ParProb3ppOscillation::load_state(
    cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision> &to_load,
    bool init)
#endif
{
  if (init) {
    to_load.setCosineList(costheta_bins);
    to_load.setEnergyList(Ebins);
    to_load.setDensity(
        {0, 1220, 3480, 5701, 6371},
        {13 * .936, 13 * .936, 11.3 * .936, 5 * .994, 3.3 * .994});
    to_load.setProductionHeight(15.0);
  }

  to_load.setMNSMatrix(asin(sqrt(GetT12())), asin(sqrt(GetT13())),
                       asin(sqrt(GetT23())), GetDeltaCP());
  to_load.setNeutrinoMasses(GetDM21sq(), GetDM32sq());
}
