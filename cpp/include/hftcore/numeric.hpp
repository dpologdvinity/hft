#pragma once

#include <cmath>

namespace hftcore {

// CPython 3.12+ builtin sum() over floats: Neumaier compensated summation,
// adding the compensation only when it is nonzero and finite. Matching it is
// required for bit-identical bar VWAPs and features.
class PySum {
 public:
  void add(double x) noexcept {
    const double t = total_ + x;
    if (std::fabs(total_) >= std::fabs(x)) {
      compensation_ += (total_ - t) + x;
    } else {
      compensation_ += (x - t) + total_;
    }
    total_ = t;
  }
  double value() const noexcept {
    return compensation_ != 0 && std::isfinite(compensation_) ? total_ + compensation_ : total_;
  }

 private:
  double total_ = 0.0;
  double compensation_ = 0.0;
};

}  // namespace hftcore
