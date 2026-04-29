#include "ParProb3ppOscillation.h"
#include "constants.h"

#include <cudaprob3/calculator_factory.hpp>
#include <cudaprob3/density_model.hpp>
#include <cudaprob3/grid.hpp>
#include <cudaprob3/oscillation_params.hpp>
#include <cudaprob3/types.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <print>
#include <stdexcept>

ParProb3ppOscillation::ParProb3ppOscillation(
    const std::vector<oscillaton_calc_precision> &Ebin,
    const std::vector<oscillaton_calc_precision> &costhbin)
    : Ebins(Ebin), costheta_bins(costhbin) {

  std::vector<double> cosines(costhbin.size());
  std::vector<double> energies(Ebin.size());
  for (size_t i = 0; i < costhbin.size(); ++i)
    cosines[i] = static_cast<double>(costhbin[i]);
  for (size_t i = 0; i < Ebin.size(); ++i)
    energies[i] = static_cast<double>(Ebin[i]);

  auto grid = std::make_shared<cudaprob3::ArbitraryGrid>(
      std::move(cosines), std::move(energies), 15.0);

  auto model = cudaprob3::PREMModel::fromVectors(
      {0.0, 1220.0, 3480.0, 5701.0, 6371.0},
      {13.0 * .936, 13.0 * .936, 11.3 * .936, 5.0 * .994, 3.3 * .994});
  if (!model) {
    throw std::runtime_error("Failed to build PREM model: " + model.error());
  }
  auto model_ptr = std::make_shared<cudaprob3::PREMModel>(std::move(*model));

  cudaprob3::SingleGPUCalculator::Config cfg{0, false};

  auto calc_nu = cudaprob3::SingleGPUCalculator::create(cfg, grid, model_ptr);
  if (!calc_nu) {
    throw std::runtime_error("Failed to create neutrino calculator: " +
                             calc_nu.error());
  }
  calculator_neutrino_ =
      std::make_shared<cudaprob3::SingleGPUCalculator>(std::move(*calc_nu));

  auto calc_anu = cudaprob3::SingleGPUCalculator::create(cfg, grid, model_ptr);
  if (!calc_anu) {
    throw std::runtime_error("Failed to create antineutrino calculator: " +
                             calc_anu.error());
  }
  calculator_antineutrino_ =
      std::make_shared<cudaprob3::SingleGPUCalculator>(std::move(*calc_anu));

  OscillationParameters default_params;
  re_calculate(default_params);
}

ParProb3ppOscillation::ParProb3ppOscillation(
    const ParProb3ppOscillation &from)
    : Ebins(from.Ebins), costheta_bins(from.costheta_bins),
      calculator_neutrino_(from.calculator_neutrino_),
      calculator_antineutrino_(from.calculator_antineutrino_) {}

ParProb3ppOscillation &
ParProb3ppOscillation::operator=(const ParProb3ppOscillation &from) {
  if (this != &from) {
    Ebins = from.Ebins;
    costheta_bins = from.costheta_bins;
    calculator_neutrino_ = from.calculator_neutrino_;
    calculator_antineutrino_ = from.calculator_antineutrino_;
  }
  return *this;
}

ParProb3ppOscillation::~ParProb3ppOscillation() = default;

void ParProb3ppOscillation::re_calculate(const OscillationParameters &p) {
  cudaprob3::OscillationParams params(
      std::asin(std::sqrt(p.GetT12())),
      std::asin(std::sqrt(p.GetT13())),
      std::asin(std::sqrt(p.GetT23())),
      p.GetDeltaCP(),
      p.GetDM21sq(), p.GetDM32sq());

  auto result_nu =
      calculator_neutrino_->calculate(params, cudaprob3::NeutrinoType::Neutrino);
  if (!result_nu) {
    throw std::runtime_error("Neutrino calculation failed: " + result_nu.error());
  }

  auto result_anu =
      calculator_antineutrino_->calculate(params,
                                          cudaprob3::NeutrinoType::Antineutrino);
  if (!result_anu) {
    throw std::runtime_error("Antineutrino calculation failed: " +
                             result_anu.error());
  }
}

std::array<std::array<double, 3>, 3>
ParProb3ppOscillation::GetProb(int flavor, double E, double costheta,
                               const OscillationParameters &p) const {
  throw std::runtime_error("Not implemented");
}

[[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
ParProb3ppOscillation::GetProb_Hists(const std::vector<double> &Ebin,
                                     const std::vector<double> &costhbin,
                                     const OscillationParameters &p) {
  re_calculate(p);

  auto result_nu = cudaprob3::ResultView{
      calculator_neutrino_->rawResults(),
      calculator_neutrino_->nCosines(),
      calculator_neutrino_->nEnergies()};
  auto result_anu = cudaprob3::ResultView{
      calculator_antineutrino_->rawResults(),
      calculator_antineutrino_->nCosines(),
      calculator_antineutrino_->nEnergies()};

  std::array<std::array<std::array<TH2D, 2>, 2>, 2> ret{};
  constexpr auto type_matrix = std::to_array(
      {std::to_array({cudaprob3::ProbType::e_e, cudaprob3::ProbType::e_m}),
       std::to_array({cudaprob3::ProbType::m_e, cudaprob3::ProbType::m_m})});

  for (int i = 0; i < 2; ++i) {
    auto &this_result = i == 0 ? result_nu : result_anu;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        auto &this_prob = ret[i][j][k];
        auto &this_term = type_matrix[j][k];
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
                this_result.probability(out_hist_costh_index, energy_bin_index,
                                        this_term));
          }
        }
      }
    }
  }
  return ret;
}

[[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
ParProb3ppOscillation::GetProb_Hists_3F(const std::vector<double> &Ebin,
                                        const std::vector<double> &costhbin,
                                        const OscillationParameters &p) {
  re_calculate(p);

  auto result_nu = cudaprob3::ResultView{
      calculator_neutrino_->rawResults(),
      calculator_neutrino_->nCosines(),
      calculator_neutrino_->nEnergies()};
  auto result_anu = cudaprob3::ResultView{
      calculator_antineutrino_->rawResults(),
      calculator_antineutrino_->nCosines(),
      calculator_antineutrino_->nEnergies()};

  std::array<std::array<std::array<TH2D, 3>, 3>, 2> ret{};
  constexpr auto type_matrix = std::to_array(
      {std::to_array({cudaprob3::ProbType::e_e, cudaprob3::ProbType::e_m,
                      cudaprob3::ProbType::e_t}),
       std::to_array({cudaprob3::ProbType::m_e, cudaprob3::ProbType::m_m,
                      cudaprob3::ProbType::m_t}),
       std::to_array({cudaprob3::ProbType::t_e, cudaprob3::ProbType::t_m,
                      cudaprob3::ProbType::t_t})});

  for (int i = 0; i < 2; ++i) {
    auto &this_result = i == 0 ? result_nu : result_anu;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        auto &this_prob = ret[i][j][k];
        auto &this_term = type_matrix[j][k];
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
                this_result.probability(out_hist_costh_index, energy_bin_index,
                                        this_term));
          }
        }
      }
    }
  }
  return ret;
}
