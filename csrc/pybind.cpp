#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "makelevelset3.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

py::array_t<float> compute(py::array_t<float> vertices,
                           py::array_t<unsigned int> faces, int size_x, int size_y, int size_z) {
  // input
  std::vector<Vec3f> V;
  for (int i = 0; i < vertices.shape(0); ++i) {
    V.push_back(Vec3f(vertices.at(i, 0), vertices.at(i, 1), vertices.at(i, 2)));
  }
  std::vector<Vec3ui> F;
  for (int i = 0; i < faces.shape(0); ++i) {
    F.push_back(Vec3ui(faces.at(i, 0), faces.at(i, 1), faces.at(i, 2)));
  }

  // bounding box
  Vec3f bbmin(-1.0f, -1.0f, -1.0f);
  Vec3f bbmax(1.0f, 1.0f, 1.0f);
  float dx = 2.0f / (float)size_x;
  float dy = 2.0f / (float)size_y;
  float dz = 2.0f / (float)size_z;

  // compute level sets
  Array3f grid;
  make_level_set3(F, V, bbmin, dx, dy, dz, size_x, size_y, size_z, grid);

  // output
  py::array_t<float> sdf({size_x, size_y, size_z});
  for (int x = 0; x < size_x; x++) {
    for (int y = 0; y < size_y; y++) {
      for (int z = 0; z < size_z; z++) {
        sdf.mutable_at(x, y, z) = grid(x, y, z);
      }
    }
  }
  return sdf;
}

PYBIND11_MODULE(core, m) {
  m.def("compute", &compute, R"pbdoc(
        Compute the SDF from an input mesh.

        Args:
          vertices (np.ndarray): The vertex array with shape (Nv, 3), and
              vertices MUST be in range [-1, 1].
          faces (np.ndarray): The face array with shape (Nf, 3).
          size_x (int): The x resolution of resulting SDF.
          size_y (int): The y resolution of resulting SDF.
          size_z (int): The z resolution of resulting SDF.
        )pbdoc",
        py::arg("vertices"), py::arg("faces"), py::arg("size_x") = 128, py::arg("size_y") = 128, py::arg("size_z") = 128);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
