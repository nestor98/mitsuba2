#include <mitsuba/core/string.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SpecularDiffusePolarized final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, MicrofacetDistribution)

    SpecularDiffusePolarized(const Properties &props) : Base(props) {
        m_diffuse_reflectance  = props.texture<Texture>("diffuse_reflectance",  .5f);

        if (props.has_property("specular_reflectance"))
            m_specular_reflectance = props.texture<Texture>("specular_reflectance", 1.f);

        /// Specifies the internal index of refraction at the interface
        ScalarFloat int_ior = lookup_ior(props, "int_ior", "polypropylene");

        /// Specifies the external index of refraction at the interface
        ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air");

        if (int_ior < 0.f || ext_ior < 0.f || int_ior == ext_ior)
            Throw("The interior and exterior indices of "
                  "refraction must be positive and differ!");

        m_eta = int_ior / ext_ior;

        mitsuba::MicrofacetDistribution<ScalarFloat, Spectrum> distr(props);
        m_type = distr.type();
        m_sample_visible = distr.sample_visible();

        m_alpha_u = distr.alpha_u();
        m_alpha_v = distr.alpha_v();

        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::DiffuseReflection;
        if (m_alpha_u != m_alpha_v)
            m_flags = m_flags | BSDFFlags::Anisotropic;
        ek::set_attr(this, "flags", m_flags);
        m_flags = m_flags | BSDFFlags::FrontSide;

        m_components.clear();
        m_components.push_back(m_flags);

        parameters_changed();
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs = ek::zero<BSDFSample3f>();
        if (unlikely((!has_specular && !has_diffuse) || ek::none_or<false>(active)))
            return { bs, 0.f };

        Float prob_specular = m_specular_sampling_weight;
        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;

        Mask sample_specular = active && (sample1 < prob_specular),
             sample_diffuse  = active && !sample_specular;

        bs.eta = 1.f;

        if (ek::any_or<true>(sample_specular)) {
            MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);
            Normal3f m = std::get<0>(distr.sample(si.wi, sample2));

            ek::masked(bs.wo, sample_specular) = reflect(si.wi, m);
            ek::masked(bs.sampled_component, sample_specular) = 0;
            ek::masked(bs.sampled_type, sample_specular) = +BSDFFlags::GlossyReflection;
        }

        if (ek::any_or<true>(sample_diffuse)) {
            ek::masked(bs.wo, sample_diffuse) = warp::square_to_cosine_hemisphere(sample2);
            ek::masked(bs.sampled_component, sample_diffuse) = 1;
            ek::masked(bs.sampled_type, sample_diffuse) = +BSDFFlags::DiffuseReflection;
        }

        bs.pdf = pdf(ctx, si, bs.wo, active);
        active &= bs.pdf > 0.f;
        Spectrum result = eval(ctx, si, bs.wo, active);

        return { bs, result / bs.pdf & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;
        if (unlikely((!has_specular && !has_diffuse) || ek::none_or<false>(active)))
            return 0.f;

        Spectrum result = 0.f;

        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to the coordinate system rotations for polarization-aware
               pBSDFs below we need to know the propagation direction of light.
               In the following, light arrives along `-wo_hat` and leaves along
               `+wi_hat`. */
            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : si.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? si.wi : wo;

            if (has_specular) {
                MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);
                Vector3f H = ek::normalize(wo + si.wi);
                Float D = distr.eval(H);

                // Mueller matrix for specular reflection.
                Spectrum F = mueller::specular_reflection(Frame3f::cos_theta(wo_hat), m_eta);

                /* The Stokes reference frame vector of this matrix lies perpendicular
                   to the plane of reflection. */
                Vector3f s_axis_in  = ek::normalize(ek::cross(H, -wo_hat)),
                         s_axis_out = ek::normalize(ek::cross(H, wi_hat));

                /* Rotate in/out reference vector of F s.t. it aligns with the implicit
                   Stokes bases of -wo_hat & wi_hat. */
                F = mueller::rotate_mueller_basis(F,
                                                  -wo_hat, s_axis_in,  mueller::stokes_basis(-wo_hat),
                                                   wi_hat, s_axis_out, mueller::stokes_basis(wi_hat));

                Float G = distr.G(si.wi, wo, H);
                Float value = D * G / (4.f * cos_theta_i);

                UnpolarizedSpectrum spec = m_specular_reflectance ? m_specular_reflectance->eval(si, active)
                                                                  : 1.f;
                result += spec * F * value;
            }

            if (has_diffuse) {
                /* Diffuse scattering is modeled a a sequence of events:
                   1) Specular refraction inside
                   2) Subsurface scattering
                   3) Specular refraction outside again
                   where both refractions reduce the energy of the diffuse component.
                   The refraction to the outside will introduce some polarization. */

                // Refract inside
                Spectrum To = mueller::specular_transmission(ek::abs(Frame3f::cos_theta(wo_hat)), m_eta);

                // Diffuse subsurface scattering.
                Spectrum diff = mueller::depolarizer(m_diffuse_reflectance->eval(si, active));

                // Refract outside
                Normal3f n(0.f, 0.f, 1.f);
                Float inv_eta = ek::rcp(m_eta);
                Float cos_theta_t_i = std::get<1>(fresnel(cos_theta_i, m_eta));
                Vector3f wi_hat_p = -refract(wi_hat, cos_theta_t_i, inv_eta);
                Spectrum Ti = mueller::specular_transmission(ek::abs(Frame3f::cos_theta(wi_hat_p)), inv_eta);

                diff = Ti * diff * To;

                /* The Stokes reference frame vector of `diff` lies perpendicular
                   to the plane of reflection. */
                Vector3f s_axis_in  = ek::normalize(ek::cross(n, -wo_hat)),
                         s_axis_out = ek::normalize(ek::cross(n,  wi_hat));

                /* Rotate in/out reference vector of `diff` s.t. it aligns with the
                   implicit Stokes bases of -wo_hat & wi_hat. */
                diff = mueller::rotate_mueller_basis(diff,
                                                      -wo_hat, s_axis_in,  mueller::stokes_basis(-wo_hat),
                                                       wi_hat, s_axis_out, mueller::stokes_basis(wi_hat));

                result += diff * ek::InvPi<Float> * cos_theta_o;
            }
        } else {
            if (has_specular) {
                MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);
                Vector3f H = ek::normalize(wo + si.wi);
                Float D = distr.eval(H);

                Spectrum F = std::get<0>(fresnel(dot(si.wi, H), m_eta));
                Float G = distr.G(si.wi, wo, H);
                Float value = D * G / (4.f * cos_theta_i);

                UnpolarizedSpectrum spec = m_specular_reflectance ? m_specular_reflectance->eval(si, active)
                                                                  : 1.f;
                result += spec * F * value;
            }

            if (has_diffuse) {
                UnpolarizedSpectrum diff = m_diffuse_reflectance->eval(si, active);
                /* Diffuse scattering is modeled a a sequence of events:
                   1) Specular refraction inside
                   2) Subsurface scattering
                   3) Specular refraction outside again
                   where both refractions reduce the energy of the diffuse component. */
                UnpolarizedSpectrum r_i = std::get<0>(fresnel(cos_theta_i, m_eta)),
                                    r_o = std::get<0>(fresnel(cos_theta_o, m_eta));
                diff = (1.f - r_o) * diff * (1.f - r_i);
                result += diff * ek::InvPi<Float> * cos_theta_o;
            }
        }

        return result & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || ek::none_or<false>(active)))
            return 0.f;

        Float prob_specular = m_specular_sampling_weight,
              prob_diffuse  = 1.f - prob_specular;
        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;

        // Specular component
        Vector3f H = ek::normalize(wo + si.wi);
        MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);

        Float p_specular;
        if (m_sample_visible)
            p_specular = distr.eval(H) * distr.smith_g1(si.wi, H) / (4.f * cos_theta_i);
        else
            p_specular = distr.pdf(si.wi, H) / (4.f * ek::dot(wo, H));
        ek::masked(p_specular, ek::dot(si.wi, H) <= 0.f || ek::dot(wo, H) <= 0.f) = 0.f;

        // Diffuse component
        Float p_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);

        return prob_specular * p_specular + prob_diffuse * p_diffuse;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("diffuse_reflectance", m_diffuse_reflectance.get());
        if (m_specular_reflectance)
            callback->put_object("specular_reflectance", m_specular_reflectance.get());
        if (!has_flag(m_flags, BSDFFlags::Anisotropic))
            callback->put_parameter("alpha", m_alpha_u);
        else {
            callback->put_parameter("alpha_u", m_alpha_u);
            callback->put_parameter("alpha_v", m_alpha_v);
        }
        callback->put_parameter("eta", m_eta);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        /* Compute weights that further steer samples towards
           the specular or diffuse components */
        Float d_mean = m_diffuse_reflectance->mean(),
              s_mean = 1.f;

        if (m_specular_reflectance)
            s_mean = m_specular_reflectance->mean();

        m_specular_sampling_weight = s_mean / (d_mean + s_mean);

        ek::eval(m_eta, m_alpha_u, m_alpha_v, m_specular_sampling_weight);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SpecularDiffusePolarized[" << std::endl
            << "  diffuse_reflectance = " << string::indent(m_diffuse_reflectance) << "," << std::endl;
        if (m_specular_reflectance)
           oss << "  specular_reflectance = " << string::indent(m_specular_reflectance) << "," << std::endl;
        oss << "  distribution = " << m_type << "," << std::endl
            << "  sample_visible = " << m_sample_visible << "," << std::endl
            << "  alpha_u = " << m_alpha_u << "," << std::endl
            << "  alpha_v = " << m_alpha_v << "," << std::endl
            << "  eta = " << m_eta << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Diffuse reflectance component
    ref<Texture> m_diffuse_reflectance;
    /// Specular reflectance component
    ref<Texture> m_specular_reflectance;

    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Importance sample the distribution of visible normals?
    bool m_sample_visible;
    /// Roughness value
    Float m_alpha_u, m_alpha_v;

    /// Relative refractive index
    Float m_eta;

    /// Sampling weight for specular component
    Float m_specular_sampling_weight;
};

MTS_IMPLEMENT_CLASS_VARIANT(SpecularDiffusePolarized, BSDF)
MTS_EXPORT_PLUGIN(SpecularDiffusePolarized, "Specular-diffuse polarized")
NAMESPACE_END(mitsuba)
