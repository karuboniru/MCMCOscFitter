#pragma once
#include <concepts>

namespace mcmc_concepts {

template <typename T>
concept Proposable = requires(T& t) {
    { t.proposeStep() } -> std::same_as<void>;
};

template <typename T>
concept HasLogLikelihood = requires(const T& t) {
    { t.GetLogLikelihood() } -> std::same_as<double>;
};

template <typename T>
concept MCMCState = Proposable<T> && HasLogLikelihood<T>;

} // namespace mcmc_concepts
