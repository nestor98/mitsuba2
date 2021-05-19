#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>


#include "D:\Universidad\cuarto-2\TFG\TFG-SDF\sdf.h"
//#include "D:\Universidad\cuarto-2\TFG\TFG-SDF\src\primitives\examples\displaced-sphere.hpp"

//#include "D:\Universidad\cuarto-2\TFG\TFG-SDF\src\sphere-marcher\sphere-marcher.hpp"



#if defined(MTS_ENABLE_OPTIX)
    #include "optix/SDF.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-SDF:

SDF (:monosp:`SDF`)
-------------------------------------------------

.. pluginparameters::


 * - basesdf
   - |string|
   - Base SDF (Default: sphere)
 * - center
   - |point|
   - Center of the SDF (Default: (0, 0, 0))
 * - radius
   - |float|
   - Radius of the SDF (Default: 1)
 * - flip_normals
   - |bool|
   - Is the SDF inverted, i.e. should the normal vectors be flipped? (Default:|false|, i.e.
     the normals point outside)
 * - to_world
   - |transform|
   -  Specifies an optional linear object-to-world transformation.
      Note that non-uniform scales and shears are not permitted!
      (Default: none, i.e. object space = world space)


.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_SDF_basic.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_SDF_parameterization.jpg
   :caption: A textured SDF with the default parameterization
.. subfigend::
   :label: fig-SDF

This shape plugin describes a simple SDF intersection primitive. It should
always be preferred over SDF approximations modeled using triangles.

A SDF can either be configured using a linear :monosp:`to_world` transformation or the :monosp:`center` and :monosp:`radius` parameters (or both).
The two declarations below are equivalent.

.. code-block:: xml

    <shape type="SDF">
        <transform name="to_world">
            <scale value="2"/>
            <translate x="1" y="0" z="0"/>
        </transform>
        <bsdf type="diffuse"/>
    </shape>

    <shape type="SDF">
        <point name="center" x="1" y="0" z="0"/>
        <float name="radius" value="2"/>
        <bsdf type="diffuse"/>
    </shape>

When a :ref:`SDF <shape-SDF>` shape is turned into an :ref:`area <emitter-area>`
light source, Mitsuba 2 switches to an efficient
`sampling strategy <https://www.akalin.com/sampling-visible-SDF>`_ by Fred Akalin that
has particularly low variance.
This makes it a good default choice for lighting new scenes.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_SDF_light_mesh.jpg
   :caption: Spherical area light modeled using triangles
.. subfigure:: ../../resources/data/docs/images/render/shape_SDF_light_analytic.jpg
   :caption: Spherical area light modeled using the :ref:`SDF <shape-SDF>` plugin
.. subfigend::
   :label: fig-SDF-light
 */

template <typename Float, typename Spectrum>
class SDFWrapper final : public Shape<Float, Spectrum> {
private:

    using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
    using Double3 = Vector<Double, 3>;

    using float3 = std::array<float, 3>;

    // convert Double3 to float3
    float3 toFloat3(const Double3& v) const {
        float3 res;
        for (int i = 0; i < 3; i++) {
            res[i] = float(v[i]);
        }
        return res;
    }

     // convert float3 to normal
    mitsuba::Normal<float,3> toNormal(const float3 &v) const {
        //return mitsuba::Normal<float,3>(v[0], v[1], v[2]);

        mitsuba::Normal<float,3> n(v[0], v[1], v[2]);
        //std::cout << n << "\n";
        return n;
    }

public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    SDFWrapper(const Properties &props) : Base(props), marcher(100000, 10000.0f, 1e-3) {
        /// Are the SDF normals pointing inwards? default: no
        m_flip_normals = props.bool_("flip_normals", false);

        //std::cout << "props: " << props << "\n";
        if (props.has_property("basesdf"))
            m_basesdf = props.string("basesdf");
        // Update the to_world transform if radius and center are also provided
        m_to_world = m_to_world * ScalarTransform4f::translate(props.point3f("center", 0.f));
        float r = (props.has_property("radius")) ? props.float_("radius", 1.f) : 1;
        //m_to_world = m_to_world * ScalarTransform4f::scale(r);
        m_radius = r;
        // Sphere --------------------------------
        sdf = sdf::Sphere(float3{ 0.f, 0.f, 0.f }, r);
        
        // Disp:
        //sdf = sdf::Displacement(sdf::Sphere(float3{ 0.f, 0.f, 0.f }, r), sdf::SineSDF(0, 5.0/r, 0.2*r));
        /* Union --------------------------------
        sdf = sdf::Union(sdf::Sphere({ -1.5f * r, 0, 0 }, r),
                   sdf::Sphere({ 0, 0, 0 }, r),
                   sdf::Sphere({ +1.5f * r, 0, 0 }, r));
        */
        //std::cout << props.
        if (props.has_property("basesdf")) {
            //sdf = sdf::from_commandline(std::vector<std::string>{std::string("-sdf"), props.string("basesdf"), "-r", std::to_string(m_radius)});
            std::vector<std::string> args{
                std::string("-sdf"),
                props.string("basesdf"),
            };
            if (props.has_property("iterations")) // fractals, etc
                args.emplace_back(std::to_string(props.int_("iterations")));
            std::vector<std::string> possible_mods{"rotate-x","rotate-z", "rotate-y", "scale", "move", "mirror"};
            for (const auto &mod : possible_mods) {
                if (props.has_property(mod)) {
                    args.emplace_back("-mod");
                    args.emplace_back(mod);
                    args.emplace_back(props.string(mod));
                } 
            }
            std::string disp = "sine-displacement";
            if (props.has_property(disp)) {
                args.emplace_back("-mod");
                args.emplace_back(disp);
                props.mark_queried(disp);
                std::vector<std::string> disp_args{ "-f", "-a","-c"};
                for (const auto &arg : disp_args) {
                    if (props.has_property(arg)) {
                        args.emplace_back(arg);
                        args.emplace_back(props.string(arg));
                    }
                }
            }

            sdf = sdf::from_commandline(args);
        } 
        
        if (props.has_property("modification") &&
            props.string("modification") == std::string("sinedisplacement")) // TODO: args in xml
            sdf = sdf::Displacement(sdf, sdf::SineSDF(0, 5.0 / r, 0.1 * r));


        //sdf->setNormalEpsilon(r * 1e-4);
        //sdf->setIntersectionEpsilon(r * 1e-2);
        update();
        set_children();
    }

    void update() {
        // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
            abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");

        if (!(abs(S[0][0] - S[1][1]) < 1e-6f && abs(S[0][0] - S[2][2]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling!");

        m_center = T;
        //m_radius = S[0][0];

        if (m_radius <= 0.f) {
            m_radius = std::abs(m_radius);
            m_flip_normals = !m_flip_normals;
        }

        // Reconstruct the to_world transform with uniform scaling and no shear
        //m_to_world  = transform_compose(ScalarMatrix3f(m_radius), Q, T);
        m_to_world = transform_compose(ScalarMatrix3f(1.0f), Q, T); // no scaling for sdfs!
        m_to_object = m_to_world.inverse();

        m_inv_surface_area = rcp(surface_area());
    }

    
    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        // Temporary:
        bbox.min = -std::numeric_limits<float>::infinity(); // m_center - m_radius*5000.0f;
        bbox.max = std::numeric_limits<float>::infinity();// m_center +
                                                          // m_radius*5000.0f;
        // ...
        return bbox;
    }

    ScalarFloat surface_area() const override {
        return 4.f * math::Pi<ScalarFloat> * m_radius * m_radius;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        std::cout << "sample_position\n";

        Point3f local = warp::square_to_uniform_sphere(sample);

        PositionSample3f ps;
        ps.p = fmadd(local, m_radius, m_center);
        ps.n = local;

        if (m_flip_normals)
            ps.n = -ps.n;

        ps.time = time;
        ps.delta = m_radius == 0.f;
        ps.pdf = m_inv_surface_area;

        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_inv_surface_area;
    }

    // For emitters:
    DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                       Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        DirectionSample3f result = zero<DirectionSample3f>();
        //std::cout<< "sample_direction\n";

        Vector3f dc_v = m_center - it.p;
        Float dc_2 = squared_norm(dc_v);

        Float radius_adj = m_radius * (m_flip_normals ? (1.f + math::RayEpsilon<Float>) :
                                                        (1.f - math::RayEpsilon<Float>));
        Mask outside_mask = active && dc_2 > sqr(radius_adj);
        if (likely(any(outside_mask))) {
            Float inv_dc            = rsqrt(dc_2),
                  sin_theta_max     = m_radius * inv_dc,
                  sin_theta_max_2   = sqr(sin_theta_max),
                  inv_sin_theta_max = rcp(sin_theta_max),
                  cos_theta_max     = safe_sqrt(1.f - sin_theta_max_2);

            /* Fall back to a Taylor series expansion for small angles, where
               the standard approach suffers from severe cancellation errors */
            Float sin_theta_2 = select(sin_theta_max_2 > 0.00068523f, /* sin^2(1.5 deg) */
                                       1.f - sqr(fmadd(cos_theta_max - 1.f, sample.x(), 1.f)),
                                       sin_theta_max_2 * sample.x()),
                  cos_theta = safe_sqrt(1.f - sin_theta_2);

            // Based on https://www.akalin.com/sampling-visible-SDF
            Float cos_alpha = sin_theta_2 * inv_sin_theta_max +
                              cos_theta * safe_sqrt(fnmadd(sin_theta_2, sqr(inv_sin_theta_max), 1.f)),
                  sin_alpha = safe_sqrt(fnmadd(cos_alpha, cos_alpha, 1.f));

            auto [sin_phi, cos_phi] = sincos(sample.y() * (2.f * math::Pi<Float>));

            Vector3f d = Frame3f(dc_v * -inv_dc).to_world(Vector3f(
                cos_phi * sin_alpha,
                sin_phi * sin_alpha,
                cos_alpha));

            DirectionSample3f ds = zero<DirectionSample3f>();
            ds.p        = fmadd(d, m_radius, m_center);
            ds.n        = d;
            ds.d        = ds.p - it.p;

            Float dist2 = squared_norm(ds.d);
            ds.dist     = sqrt(dist2);
            ds.d        = ds.d / ds.dist;
            ds.pdf      = warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_theta_max);
            masked(ds.pdf, ds.dist == 0.f) = 0.f;

            result[outside_mask] = ds;
        }

        Mask inside_mask = andnot(active, outside_mask);
        if (unlikely(any(inside_mask))) {
            Vector3f d = warp::square_to_uniform_sphere(sample);
            DirectionSample3f ds = zero<DirectionSample3f>();
            ds.p        = fmadd(d, m_radius, m_center);
            ds.n        = d;
            ds.d        = ds.p - it.p;

            Float dist2 = squared_norm(ds.d);
            ds.dist     = sqrt(dist2);
            ds.d        = ds.d / ds.dist;
            ds.pdf      = m_inv_surface_area * dist2 / abs_dot(ds.d, ds.n);

            result[inside_mask] = ds;
        }

        result.time = it.time;
        result.delta = m_radius == 0.f;

        if (m_flip_normals)
            result.n = -result.n;

        return result;
    }

    // Also for emitters
    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        //std::cout << "pdf_direction\n";

        // Sine of the angle of the cone containing the SDF as seen from 'it.p'.
        Float sin_alpha = m_radius * rcp(norm(m_center - it.p)),
              cos_alpha = enoki::safe_sqrt(1.f - sin_alpha * sin_alpha);

        return select(sin_alpha < math::OneMinusEpsilon<Float>,
            // Reference point lies outside the SDF
            warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_alpha),
            m_inv_surface_area * sqr(ds.dist) / abs_dot(ds.d, ds.n)
        );
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);


        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double3 o = Double3(ray.o) - Double3(m_center);
        Double3 d(ray.d);

        // Sphere marching intersection:
        //std::cout << "Vamos a trazar esferas\n";
        auto is = marcher.trace(sdf, toFloat3(o), toFloat3(d));
        Mask solution_found = bool(is) && is->distance>=mint && is->distance>0;

        active &= solution_found;// && !out_bounds && !in_bounds;
        

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t = select(active,
                      is->distance,
                      math::Infinity<Float>);
        pi.shape = this;

        //if (is)
          //  std::cout << "PreliminaryIntersection: " << pi.t << "\n";

        return pi;
    }

    Mask ray_test(const Ray3f &ray, Mask active) const override {
        MTS_MASK_ARGUMENT(active);


        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double3 o = Double3(ray.o) - Double3(m_center);
        Double3 d(ray.d);

        // Sphere marching intersection:
        auto is = marcher.trace(sdf, toFloat3(o), toFloat3(d));
        Mask solution_found = bool(is) && is->distance >= mint && is->distance > 0; // && is->distance > 0;
        //std::cout << "RayTest: " << solution_found << "\n";

        return solution_found && active; // !out_bounds && !in_bounds;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);
        // Point for normal
        auto p = ray(pi.t);//fmadd(ray.d, si.t, ray.o);//ray.o + ray.d*si.t;
        auto pLocal = m_to_object.transform_affine(p);
        // Normal
        // si.sh_frame.n = normalize(ray(pi.t) - m_center);
        si.sh_frame.n = normalize(toNormal(sdf.normal(toFloat3(pLocal))));

        // separate point from surface?:
        float eps = marcher.getEps() * 200.0f; // m_radius/1000.0;
        si.p = fmadd(si.sh_frame.n, eps, p);
        //si.p      = p;
        if (likely(has_flag(flags, HitComputeFlags::UV))) {
            // std::cout << "UV not implemented in SDF!" << '\n';
            Vector3f local = m_to_object.transform_affine(si.p);
            //
            Float rd_2  = sqr(local.x()) + sqr(local.y()),
                  theta = unit_angle_z(local),
                  phi   = atan2(local.y(), local.x());

            masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

            si.uv = Point2f(phi * math::InvTwoPi<Float>, theta * math::InvPi<Float>);
            if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {
                // std::cout << "dPdUV in SDF!" << '\n';
                si.dp_du = Vector3f(-local.y(), local.x(), 0.f);

                Float rd      = sqrt(rd_2),
                      inv_rd  = rcp(rd),
                      cos_phi = local.x() * inv_rd,
                      sin_phi = local.y() * inv_rd;

                si.dp_dv = Vector3f(local.z() * cos_phi,
                                    local.z() * sin_phi,
                                    -rd);

                Mask singularity_mask = active && eq(rd, 0.f);
                if (unlikely(any(singularity_mask)))
                    si.dp_dv[singularity_mask] = Vector3f(1.f, 0.f, 0.f);

                si.dp_du = m_to_world * si.dp_du * (2.f * math::Pi<Float>);
                si.dp_dv = m_to_world * si.dp_dv * math::Pi<Float>;
            }
        }

        if (m_flip_normals)
            si.sh_frame.n = -si.sh_frame.n;
        // Geometric normal:
        si.n = si.sh_frame.n;
        // << "SurfaceInteraction: " << si.n << "\n";
        //
        if (has_flag(flags, HitComputeFlags::dNSdUV)) {
            std::cout << "dNSdUV not implemented!" << '\n';
            // ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius;
            // si.dn_du = si.dp_du * inv_radius;
            // si.dn_dv = si.dp_dv * inv_radius;
        }

        if (!si.is_valid()) {
          std::cout << "invalid SI !!" << '\n';
        }

        return si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
            if (!m_optix_data_ptr

                m_optix_data_ptr = cuda_malloc(sizeof(OptixSDFData));

            OptixSDFData data = { bbox(), m_to_world, m_to_object,
                                     m_center, m_radius, m_flip_normals };

            cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixSDFData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SDF[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  center = "  << m_center << "," << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Center in world-space
    ScalarPoint3f m_center;
    /// Radius in world-space
    ScalarFloat m_radius;

    ScalarFloat m_inv_surface_area;

    bool m_flip_normals;

    std::string m_basesdf;

    sdf::SDF sdf = sdf::Sphere(); // The sdf defining the shape
    sdf::SphereMarcher marcher;
};

MTS_IMPLEMENT_CLASS_VARIANT(SDFWrapper, Shape)
MTS_EXPORT_PLUGIN(SDFWrapper, "SDF intersection primitive");
NAMESPACE_END(mitsuba)
