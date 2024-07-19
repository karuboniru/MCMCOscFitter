#pragma once

class StateI {
public:
  StateI() = default;
  StateI(const StateI &) = default;
  StateI(StateI &&) = default;
  StateI &operator=(const StateI &) = default;
  StateI &operator=(StateI &&) = default;

  virtual ~StateI() = default;

  virtual void proposeStep() = 0;
  virtual double GetLogLikelihood() const = 0;
};