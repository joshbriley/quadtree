#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double evaluate_polynomial(
    py::array_t<double> vals,
    double x,
    double y,
    py::array_t<double> bounds
) {
    auto v = vals.unchecked<2>();   // fast NumPy access
    auto b = bounds.unchecked<1>();

    double xmin = b(0);
    double xmax = b(1);
    double ymin = b(2);
    double ymax = b(3);

    // Normalize
    double tx = (x - xmin) / (xmax - xmin);
    double ty = (y - ymin) / (ymax - ymin);

    double nodes[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};

    double Lx[4] = {1,1,1,1};
    double Ly[4] = {1,1,1,1};

    // Compute Lagrange basis
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) {
                Lx[i] *= (tx - nodes[j]) / (nodes[i] - nodes[j]);
                Ly[i] *= (ty - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
    }

    // Tensor product
    double result = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result += v(i,j) * Lx[i] * Ly[j];
        }
    }

    return result;
}

// Python module name = polyinterp
PYBIND11_MODULE(polyinterp, m) {
    m.def("evaluate_polynomial", &evaluate_polynomial,
          "Evaluate tensor-product cubic interpolation");
}
