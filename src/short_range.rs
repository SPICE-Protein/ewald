//! This module contains code for the short-range component of the Coulomb force.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_CMP_LT_OQ, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_set1_ps};

use lin_alg::f32::Vec3;
// SIMD primitives only exist on x86/x86_64 (and require lin_alg's `simd` feature);
// gate the import to match the x8/x16 functions below so non-x86 targets compile.
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x16, Vec3x8, f32x16, f32x8};

use crate::INV_SQRT_PI;

///  Computes the direct, short-range component. Ideally, use a combined GPU kernel with Lennard Jones,
/// or a SIMD variant, instead of this.  We use this for short-range Coulomb forces on the CPU, as part of SPME.
/// `cutoff_dist` is the distance, in Å, at which we no longer apply any force from this component.
/// α controls the blending of short and long-range forces; 0.35Å for α is a good default for a cutoff of 10Å.
///
/// This assumes diff (and dir) is in order tgt - src.
/// Also returns potential energy. `dir` must be a unit vector.
pub fn force_coulomb_short_range(
    dir: Vec3,
    dist: f32,
    // Included in this form to share between this and Lennard Jones.
    inv_dist: f32,
    q_0: f32,
    q_1: f32,
    cutoff_dist: f32,
    α: f32,
) -> (Vec3, f32) {
    if dist >= cutoff_dist {
        return (Vec3::new_zero(), 0.);
    }

    let α_r = α * dist;

    // Fast real-space PME kernel. The old code called libm `erfc` (double) and
    // `.exp()` per pair (~30-40 ns each) — the dominant cost of the CPU
    // non-bonded loop. We use the Abramowitz & Stegun 7.1.26 identity
    //
    //   erfc(x) ≈ (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵)·exp(-x²),
    //   t = 1/(1 + p·x)
    //
    // (max |error| ≈ 1.5e-7, far tighter than MD needs) with a SINGLE exp and
    // one division per pair. Both the erfc and the exp(-x²) terms the force
    // needs come out of that one exp. Near the cutoff the force is dominated
    // by the exp term anyway, so the tiny erfc error is negligible.
    let x2 = α_r * α_r;
    let e = (-x2).exp(); // exp(-x²) — shared by erfc and the force term
    let t = 1.0 / (1.0 + 0.3275911 * α_r); // 1/(1+p·x)
    // Horner: a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵
    let poly =
        ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erfc_term = poly * e;

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist + 2.0 * α * e * INV_SQRT_PI * inv_dist);

    (dir * force_mag, energy)
}

#[cfg(target_arch = "x86_64")]
pub fn force_coulomb_short_range_x8(
    dir: Vec3x8,
    dist: f32x8,
    inv_dist: f32x8,
    q_0: f32x8,
    q_1: f32x8,
    cutoff_dist: f32x8,
    // Alternatively, we could use a normal f32 for this, and splat it in-fn.
    α: f32x8,
) -> (Vec3x8, f32x8) {
    let α_r = α * dist;
    let erfc_term = α_r.erfc();

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x8::splat(2.) * α * exp_term * f32x8::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    // This is where we diverge from the syntax of the non-SIMD variant;
    // the outside/inside cutoff.
    // per-lane mask: keep where dist < cutoff_dist, else zero
    unsafe {
        let keep = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(dist.0, cutoff_dist.0);
        let zero = _mm256_set1_ps(0.0);

        let fx = _mm256_blendv_ps(zero, (force.x).0, keep);
        let fy = _mm256_blendv_ps(zero, (force.y).0, keep);
        let fz = _mm256_blendv_ps(zero, (force.z).0, keep);
        let en = _mm256_blendv_ps(zero, energy.0, keep);

        (
            Vec3x8 {
                x: f32x8(fx),
                y: f32x8(fy),
                z: f32x8(fz),
            },
            f32x8(en),
        )
    }
}

#[cfg(target_arch = "x86_64")]
pub fn force_coulomb_short_range_x16(
    dir: Vec3x16,
    dist: f32x16,
    // Included to share between this and Lennard Jones.
    inv_dist: f32x16,
    q_0: f32x16,
    q_1: f32x16,
    cutoff_dist: f32x16,
    // Alternatively, we could use a normal f32 for this, and splat it in-fn.
    α: f32x16,
) -> (Vec3x16, f32x16) {
    let α_r = α * dist;
    let erfc_term = α_r.erfc();

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x16::splat(2.) * α * exp_term * f32x16::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    // This is where we diverge from the syntax of the non-SIMD variant;
    // the outside/inside cutoff.
    // per-lane mask: keep where dist < cutoff_dist, else zero
    unsafe {
        use core::arch::x86_64::*;
        let keep: __mmask16 = _mm512_cmp_ps_mask::<{ _CMP_LT_OQ }>(dist.0, cutoff_dist.0);

        let fx = _mm512_maskz_mov_ps(keep, (force.x).0);
        let fy = _mm512_maskz_mov_ps(keep, (force.y).0);
        let fz = _mm512_maskz_mov_ps(keep, (force.z).0);
        let en = _mm512_maskz_mov_ps(keep, energy.0);

        (
            Vec3x16 {
                x: f32x16(fx),
                y: f32x16(fy),
                z: f32x16(fz),
            },
            f32x16(en),
        )
    }
}
