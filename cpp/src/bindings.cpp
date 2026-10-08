#include <pybind11/pybind11.h>

#include <string>

#include "hftcore/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hftcore, m) {
  m.doc() = "Low-latency market-data replay and decision engine";
  m.def("version", [] { return std::string(hftcore::version()); },
        "Engine version recorded in run identities");
}
