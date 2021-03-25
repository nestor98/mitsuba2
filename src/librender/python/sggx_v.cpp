#include <mitsuba/render/fresnel.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(sggx) {
    MTS_PY_IMPORT_TYPES()
    m.def("sggx_sample_vndf",
        &sggx_sample_vndf<Float, Spectrum>,
        sh_frame, sample,


        "cos_theta_i"_a, "eta"_a, D(fresnel))
    .def("fresnel_conductor",
        &fresnel_conductor<Float>,
        "cos_theta_i"_a, "eta"_a, D(fresnel_conductor))
    .def("fresnel_polarized",
        py::overload_cast<Float, ek::Complex<Float>>(&fresnel_polarized<Float>),
        "cos_theta_i"_a, "eta"_a, D(fresnel_polarized, 2))

}
