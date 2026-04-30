#pragma once
#include "StateI.h"
#include <random>

bool MCMCAcceptState(const StateI &current, const StateI &next);

bool MCMCAcceptState(const StateI &current, const StateI &next, std::mt19937 &rng);