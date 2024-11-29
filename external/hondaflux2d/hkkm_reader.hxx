
#include <TH2D.h>

#include <array>
#include <string>

class HKKM_READER_2D {
public:
  HKKM_READER_2D(const std::string & );
  HKKM_READER_2D(const HKKM_READER_2D &) = default;
  HKKM_READER_2D(HKKM_READER_2D &&) = default;
  HKKM_READER_2D &operator=(const HKKM_READER_2D &) = default;
  HKKM_READER_2D &operator=(HKKM_READER_2D &&) = default;
  ~HKKM_READER_2D() = default;

  [[nodiscard]] TH2D &operator[](int pdg) { return hists[pdg_index(pdg)]; }

private:
  [[nodiscard]] static size_t pdg_index(int pdg);
  ///> 0: numu, 1: numubar, 2: nue, 3: nuebar
  ///> hist(logE, costh, phi)
  ///>  E in GeV
  ///>  costh in [-1, 1]
  ///>  phi in [0, 2pi]
  std::array<TH2D, 4> hists;
};