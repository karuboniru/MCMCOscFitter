#pragma once

constexpr double H_to_C = 120. / 73.;
// constexpr double H_to_C = 1.67;

constexpr double scale_factor_1y =
    (2e10 /*LS in g*/ / (12. + 1. * H_to_C /*mass of CH_x group*/) *
     6.02214076e23 /*N_A*/) *           // number of target C12
    ((365.25) * 24 * 3600 /*1 year*/) / // seconds in a year
    1e42; // unit conversion from 1e-38 cm^2 to 1e-42 m^2

constexpr double scale_factor_6y = 6. * scale_factor_1y;

using oscillaton_calc_precision = double;