#pragma once

#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/warp.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT
Normal<Float, 3> sggx_sample_vndf(const MediumInteraction<Float, Spectrum> &mi,
                             const Point<Float, 2> &sample,
                             const Vector<Float, 6> &s) {
    const size_t XX = 0, YY = 1, ZZ = 2, XY = 3, XZ = 4, YZ = 5;

    using Matrix3f = ek::Matrix<Float, 3>;
    using Vector3f = Vector<Float, 3>;


    Matrix3f m(mi.sh_frame.s, mi.sh_frame.t, mi.sh_frame.n);
    m      = ek::transpose(m);
    Matrix3f s_mat(s[XX], s[XY], s[XZ],
                   s[XY], s[YY], s[YZ],
                   s[XZ], s[YZ], s[ZZ]);
    Matrix3f s2 = m * s_mat * ek::transpose(m);
    const size_t k = 0, j = 1, i = 2;
    Float inv_sqrt_s_ii = ek::safe_rsqrt(s2(i, i));
    Float tmp = ek::safe_sqrt(s2(j, j) * s2(i, i) - s2(j, i) * s2(j, i));
    Vector3f m_k(ek::safe_sqrt(ek::abs(ek::det(s2))) / tmp, 0.f, 0.f);
    Vector3f m_j(-inv_sqrt_s_ii * (s2(k, i) * s2(j, i) - s2(k, j) * s2(i, i)) / tmp,
                inv_sqrt_s_ii * tmp, 0.f);
    Vector3f m_i = inv_sqrt_s_ii * Vector3f(s2(k, i), s2(j, i), s2(i, i));
    Vector3f uvw = warp::square_to_cosine_hemisphere(sample);
    return mi.to_world(ek::normalize(uvw.x() * m_k + uvw.y() * m_j + uvw.z() * m_i));
}

template<typename Float>
Float sggx_ndf_pdf(const Vector<Float, 3> &wm, const Vector<Float, 6> &s) {
    const size_t XX = 0, YY = 1, ZZ = 2, XY = 3, XZ = 4, YZ = 5;

    Float det_s = abs(s[XX] * s[YY] * s[ZZ] - s[XX] * s[YZ] * s[YZ] -
                      s[YY] * s[XZ] * s[XZ] - s[ZZ] * s[XY] * s[XY] +
                      2.f * s[XY] * s[XZ] * s[YZ]);
    Float den   = wm.x() * wm.x() * (s[YY] * s[ZZ] - s[YZ] * s[YZ]) +
                wm.y() * wm.y() * (s[XX] * s[ZZ] - s[XZ] * s[XZ]) +
                wm.z() * wm.z() * (s[XX] * s[YY] - s[XY] * s[XY]) +
                2.f * (wm.x() * wm.y() * (s[XZ] * s[YZ] - s[ZZ] * s[XY]) +
                       wm.x() * wm.z() * (s[XY] * s[YZ] - s[YY] * s[XZ]) +
                       wm.y() * wm.z() * (s[XY] * s[XZ] - s[XX] * s[YZ]));
    return ek::max(det_s, 0.f) * ek::safe_sqrt(det_s) / (ek::Pi<Float> * ek::sqr(den));
}

template<typename Float>
MTS_INLINE
Float sggx_projected_area(const Vector<Float, 3> &wi,
                          const Vector<Float, 6> &s) {
    const size_t XX = 0, YY = 1, ZZ = 2, XY = 3, XZ = 4, YZ = 5;

    // Computes sqrt(wi^T * S * wi)
    Float sigma2 = wi.x() * wi.x() * s[XX] +
                   wi.y() * wi.y() * s[YY] +
                   wi.z() * wi.z() * s[ZZ] +
                   2.f * (wi.x() * wi.y() * s[XY] +
                          wi.x() * wi.z() * s[XZ] +
                          wi.y() * wi.z() * s[YZ]);
    return ek::safe_sqrt(sigma2);
}


NAMESPACE_END(mitsuba)
