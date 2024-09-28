#pragma once
#include "DataSet.h"

// simple, so stateless
// no-op for proposeStep
class SimpleDataPoint : virtual public StateI {
public:
  double E{}, costheta{}, phi{};
  int flavor{};
  SimpleDataPoint() = default;
  SimpleDataPoint &operator=(const SimpleDataPoint &) = default;
  SimpleDataPoint &operator=(SimpleDataPoint &&) = default;
  SimpleDataPoint(double E_, double costheta_, double phi_, int flavor_)
      : E(E_), costheta(costheta_), phi(phi_), flavor(flavor_) {}

  SimpleDataPoint(const SimpleDataPoint &other)
      : E(other.E), costheta(other.costheta), phi(other.phi),
        flavor(other.flavor) {}
  // SimpleDataPoint(SimpleDataPoint &&other)
  //     : E(other.E), costheta(other.costheta), phi(other.phi),
  //       flavor(other.flavor) {}

  void proposeStep() override {}
  [[nodiscard]] double GetLogLikelihood() const override { return 0; }
};

using SimpleDataSet = DataSet<SimpleDataPoint>;