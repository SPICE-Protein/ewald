#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]

//! For Smooth-Particle-Mesh Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions (e.g. of the solvent) properly.
//! See the Readme for details. The API is split into
//! two main parts: A standalone function to calculate short-range force, and
//! a struct with forces methods for long-range reciprical forces.

extern crate core;

use std::f32::consts::TAU;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaStream},
    nvrtc::Ptx,
};

mod fft;

#[cfg(feature = "cuda")]
mod gpu_shared;
pub mod short_range;
use lin_alg::f32::Vec3;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
pub use short_range::*;
use statrs::function::erf::erf;

pub use crate::fft::{fft3d_c2r, fft3d_r2c};
#[cfg(feature = "cuda")]
use crate::gpu_shared::{GpuData, GpuTables, Kernels};

#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../ewald.ptx");

const SQRT_PI: f32 = 1.7724538509055159;
const INV_SQRT_PI: f32 = 1. / SQRT_PI;
const TWO_INV_SQRT_PI: f32 = 2. / SQRT_PI;

// The cardinal B-spline of order used to spread point charges to the mesh and
// to interpolate forces back. 4 is the safe default, and should probably not be changed.
// 3 is faster, but less accurate. 5+ are slow.
const SPLINE_ORDER: usize = 4;

type Complex_ = Complex<f32>;

/// Initialize this once for the application, or once per step.
/// Note:
pub struct PmeRecip {
    /// Simultatoin box-lengths in real space.
    box_dims: (f32, f32, f32),
    /// FFT planner dimensions. B‑spline interpolation X, Y, Z. These should be based on
    /// box length, and mesh spacing. nz should be even.
    plan_dims: (usize, usize, usize),
    vol: f32,
    /// A tunable variable used in the splitting between short range and long range forces.
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    pub alpha: f32,
    /// Precomputed k-vectors and B-spline deconvolution |B(k)|^2
    kx: Vec<f32>,
    ky: Vec<f32>,
    kz: Vec<f32>,
    bmod_sq_inv_x: Vec<f32>,
    bmod_sq_inv_y: Vec<f32>,
    bmod_sq_inv_z: Vec<f32>,
    /// For CPU FFTs
    planner: FftPlanner<f32>,
    /// For GPU FFTs. None if compiling with GPU support, but there is a runtime problem,
    /// e.g. no CUDA on the system running it.
    #[cfg(feature = "cuda")]
    gpu_data: Option<GpuData>,
}

// Manual Clone: rustfft's `FftPlanner` is not `Clone` (it caches FFT plans), so
// we copy the scalar/vector state and give the clone a fresh planner — rustfft
// builds plans lazily on first use, so this is cheap. Used to give each
// environment point a truly independent copy of a built (solvated+minimized)
// system, avoiding state carry-over between reused engines.
impl Clone for PmeRecip {
    fn clone(&self) -> Self {
        Self {
            box_dims: self.box_dims,
            plan_dims: self.plan_dims,
            vol: self.vol,
            alpha: self.alpha,
            kx: self.kx.clone(),
            ky: self.ky.clone(),
            kz: self.kz.clone(),
            bmod_sq_inv_x: self.bmod_sq_inv_x.clone(),
            bmod_sq_inv_y: self.bmod_sq_inv_y.clone(),
            bmod_sq_inv_z: self.bmod_sq_inv_z.clone(),
            planner: FftPlanner::new(),
            #[cfg(feature = "cuda")]
            gpu_data: None,
        }
    }
}

impl PmeRecip {
    pub fn new(
        #[cfg(feature = "cuda")] stream: Option<&Arc<CudaStream>>,
        plan_dims: (usize, usize, usize),
        l: (f32, f32, f32),
        alpha: f32,
    ) -> Self {
        assert!(plan_dims.0 >= 4 && plan_dims.1 >= 4 && plan_dims.2 >= 4);

        let vol = l.0 * l.1 * l.2;

        let kx = make_k_array(plan_dims.0, l.0);
        let ky = make_k_array(plan_dims.1, l.1);
        let kz = make_k_array(plan_dims.2, l.2);

        let bmod_sq_inv_x = spline_bmod_sq_inv_1d(plan_dims.0, SPLINE_ORDER);
        let bmod_sq_inv_y = spline_bmod_sq_inv_1d(plan_dims.1, SPLINE_ORDER);
        let bmod_sq_inv_z = spline_bmod_sq_inv_1d(plan_dims.2, SPLINE_ORDER);

        #[cfg(feature = "cuda")]
        let mut gpu_data = None;

        #[cfg(feature = "cuda")]
        if let Some(s) = stream {
            gpu_data = match CudaContext::new(0) {
                Ok(cuda_ctx) => {
                    let kernels = {
                        let module = cuda_ctx.load_module(Ptx::from_src(PTX)).unwrap();

                        let kernel_spread = module.load_function("spread_charges").unwrap();
                        let kernel_ghat = module.load_function("apply_ghat_and_grad").unwrap();
                        // let kernel_ghat = module.load_function("apply_ghat_and_compute_potential").unwrap();
                        let kernel_gather = module.load_function("gather_forces_to_atoms").unwrap();
                        let kernel_half_spectrum =
                            module.load_function("energy_half_spectrum").unwrap();

                        Kernels {
                            kernel_spread,
                            kernel_ghat,
                            kernel_gather,
                            kernel_half_spectrum,
                        }
                    };

                    let gpu_tables = {
                        let k = (&kx, &ky, &kz);
                        let bmod2 = (&bmod_sq_inv_x, &bmod_sq_inv_y, &bmod_sq_inv_z);

                        GpuTables::new(k, bmod2, s)
                    };

                    // let planner_gpu = cufft::create_gpu_plan(plan_dims, s);

                    #[cfg(feature = "cuda")]
                    let planner_gpu = fft::create_gpu_plan(plan_dims, s);

                    Some(GpuData {
                        planner_gpu,
                        gpu_tables,
                        kernels,
                    })
                }
                Err(_) => None,
            };
        }

        // Note: planner_gpu and gpu_tables will be null/None until the first run. We
        // handle it this way since the forces fns have access to the stream or Context,
        // but this construct doens't, if we keep a unified constructor.
        Self {
            box_dims: l,
            plan_dims,
            vol,
            alpha,
            kx,
            ky,
            kz,
            bmod_sq_inv_x,
            bmod_sq_inv_y,
            bmod_sq_inv_z,
            planner: FftPlanner::new(),
            #[cfg(feature = "cuda")]
            gpu_data,
        }
    }

    /// CPU charge spreading. Z as the fast/contiguous axis.
    /// We place charges on a discrete grid, then apply a 3D FFT to the grid.
    fn spread_charges(&self, pos: &[Vec3], q: &[f32], rho: &mut [f32]) {
        let (nx, ny, nz) = self.plan_dims;
        let (lx, ly, lz) = self.box_dims;

        let nynz = ny * nz; // Z-fast layout helper

        for (r, &qi) in pos.iter().zip(q.iter()) {
            let sx = r.x / lx * nx as f32;
            let sy = r.y / ly * ny as f32;
            let sz = r.z / lz * nz as f32;

            let (ix0, wx) = bspline4_weights(sx);
            let (iy0, wy) = bspline4_weights(sy);
            let (iz0, wz) = bspline4_weights(sz);

            for a in 0..SPLINE_ORDER {
                let ix = wrap(ix0 + a as isize, nx);
                let wxa = wx[a];

                for b in 0..SPLINE_ORDER {
                    let iy = wrap(iy0 + b as isize, ny);
                    let wxy = wxa * wy[b];

                    // Z-fast base for this (ix, iy)
                    let base = ix * nynz + iy * nz;

                    for c in 0..SPLINE_ORDER {
                        let iz = wrap(iz0 + c as isize, nz);
                        let idx = base + iz; // contiguous over z
                        rho[idx] += qi * wxy * wz[c];
                    }
                }
            }
        }
    }

    /// Compute reciprocal-space forces on all positions, using the CPU. Positions must be in the
    /// primary box [0,L] per axis.
    /// todo: Is there a way to exclude static targets, or must all sources be targets?
    pub fn forces(&mut self, posits: &[Vec3], q: &[f32]) -> (Vec<Vec3>, f32) {
        let (nx, ny, nz) = self.plan_dims;

        // Z is the fast dimension; keep this consistent with the CPU FFT setup, and
        // cuFFT; note that this is cuFFT's default layout.
        let n_real = nx * ny * nz;
        let nzc = nz / 2 + 1;
        let n_k = nx * ny * nzc;

        // Charge density.
        let mut rho_real = vec![0.; n_real];
        self.spread_charges(posits, q, &mut rho_real);

        // Convert spread charges to K space
        let rho = fft3d_r2c(&mut rho_real, self.plan_dims, &mut self.planner);

        let mut phi_k = vec![Complex::new(0.0, 0.0); n_k];
        let (mut energy, _virial) = self.apply_ghat_and_compute_potential(&rho, &mut phi_k, ny, nzc);

        energy += self_energy(q, self.alpha);

        // Inverse FFT (Complex -> Real) to get Scalar Potential Grid.
        // No /N normalization: ghat encodes 1/V so IFFT(ghat * rho_k) gives
        // the physical potential directly (unnormalized IFFT convention).
        let phi_real = fft3d_c2r(&mut phi_k, self.plan_dims, &mut self.planner);

        let f = gather_forces_from_potential(posits, q, &phi_real, self.plan_dims, self.box_dims);

        (f, energy as f32)
    }

    // todo: Updated dec 2025
    /// Returns (energy, virial). The virial is the isotropic scalar virial W_lr derived from
    /// the volume derivative of E_recip: W_lr = Σ_k E_k × (1 − k²/(2α²)).
    fn apply_ghat_and_compute_potential(
        &self,
        rho: &[Complex_],
        phi_k: &mut [Complex_],
        ny: usize,
        nzc: usize,
    ) -> (f64, f64) {
        let (kx, ky, kz) = (&self.kx, &self.ky, &self.kz);
        let (bx, by, bz) = (
            &self.bmod_sq_inv_x,
            &self.bmod_sq_inv_y,
            &self.bmod_sq_inv_z,
        );

        let (vol, alpha) = (self.vol, self.alpha);
        let two_alpha_sq = 2.0 * (alpha as f64) * (alpha as f64);

        let (energy_sum, virial_sum): (f64, f64) = phi_k
            .par_iter_mut()
            .zip(rho.par_iter())
            .enumerate()
            .map(|(idx, (phi, &rho_val))| {
                let izc = idx % nzc;
                let iy = (idx / nzc) % ny;
                let ix = idx / (nzc * ny);

                let kxv = kx[ix];
                let kyv = ky[iy];
                let kzv = kz[izc];

                let k2 = kxv.mul_add(kxv, kyv.mul_add(kyv, kzv * kzv));

                if k2 == 0.0 {
                    *phi = Complex::new(0.0, 0.0);
                    return (0.0, 0.0);
                }

                let b_inv2 = (bx[ix] * by[iy] * bz[izc]) as f64; // 1/|B|^2
                if !b_inv2.is_finite() || b_inv2 <= 0.0 {
                    *phi = Complex::new(0.0, 0.0);
                    return (0.0, 0.0);
                }

                let ghat = ((2.0 * TAU) as f64 / (vol as f64))
                    * (-(k2 as f64) / (4.0 * (alpha as f64) * (alpha as f64))).exp()
                    / (k2 as f64)
                    * b_inv2;

                let val = Complex::new(
                    (rho_val.re as f64 * ghat) as f32,
                    (rho_val.im as f64 * ghat) as f32,
                );
                *phi = val;

                let mut local_energy =
                    rho_val.re as f64 * val.re as f64 + rho_val.im as f64 * val.im as f64;

                if izc > 0 && izc < (nzc - 1) {
                    local_energy *= 2.0;
                }

                // Isotropic virial from volume derivative: W_k = E_k × (1 − k²/(2α²))
                let virial_local = local_energy * (1.0 - k2 as f64 / two_alpha_sq);

                (local_energy, virial_local)
            })
            .reduce(|| (0.0, 0.0), |(e1, v1), (e2, v2)| (e1 + e2, v1 + v2));

        (0.5 * energy_sum, 0.5 * virial_sum)
    }

    /// Compute reciprocal-space forces, energy, and the isotropic scalar virial on all positions,
    /// using the CPU. Positions must be in the primary box [0,L] per axis.
    /// The virial W_lr is the volume-derivative contribution to pressure: W = Σ_k E_k(1−k²/2α²).
    pub fn forces_and_virial(&mut self, posits: &[Vec3], q: &[f32]) -> (Vec<Vec3>, f32, f64) {
        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let nzc = nz / 2 + 1;
        let n_k = nx * ny * nzc;

        let mut rho_real = vec![0.; n_real];
        self.spread_charges(posits, q, &mut rho_real);

        let rho = fft3d_r2c(&mut rho_real, self.plan_dims, &mut self.planner);

        let mut phi_k = vec![Complex::new(0.0, 0.0); n_k];
        let (mut energy, virial) =
            self.apply_ghat_and_compute_potential(&rho, &mut phi_k, ny, nzc);

        energy += self_energy(q, self.alpha);

        let phi_real = fft3d_c2r(&mut phi_k, self.plan_dims, &mut self.planner);

        let f = gather_forces_from_potential(posits, q, &phi_real, self.plan_dims, self.box_dims);

        (f, energy as f32, virial)
    }
}

/// k-array for an orthorhombic cell; FFT index convention → physical wavevector.
fn make_k_array(n: usize, L: f32) -> Vec<f32> {
    let tau_div_l = TAU / L;

    let mut out = vec![0.0; n];
    let n_half = n / 2;

    for (i, out_) in out.iter_mut().enumerate() {
        // map 0..n-1 -> signed frequency bins: 0,1,2,...,n/2,-(n/2-1),..., -1
        let fi = if i <= n_half {
            i as isize
        } else {
            (i as isize) - (n as isize)
        };
        *out_ = tau_div_l * (fi as f32);
    }
    out
}

/// Exact |B(k)|^-2 for Cardinal B-splines of order m (PME deconvolution factor).
///
/// Derived from Essmann et al. (1995) eq. 4.4. The modulus is the squared magnitude
/// of the DFT of the B-spline evaluated at integer points:
///   |b_m(k)|^2 = |Σ_{j=0}^{m-2}  M_m(j+1) * exp(2πi·j·k/n)|^2
///
/// For order 4: M_4(1)=1/6, M_4(2)=4/6, M_4(3)=1/6, giving the closed form:
///   |b_4(k)|^2 = ((4 + 2·cos(2πk/n)) / 6)^2
///
/// We store 1/|b|^2 for use in the influence function G(k).
fn spline_bmod_sq_inv_1d(n: usize, m: usize) -> Vec<f32> {
    assert!(m >= 1);
    // Integer-point values of the cardinal B-spline M_m, for j = 1..m-1.
    // (M_m is zero at j=0 and j=m, so these are all non-zero entries.)
    // For m=4: [M_4(1), M_4(2), M_4(3)] = [1/6, 4/6, 1/6]
    let bspline_ints: &[f32] = match m {
        4 => &[1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0],
        3 => &[0.5, 0.5],
        5 => &[1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0],
        _ => panic!("Unsupported B-spline order {m}; add its integer-point values above."),
    };

    let mut v = vec![0.0f32; n];
    for k in 0..n {
        // |b_m(k)|^2 = |Σ_j  bspline_ints[j] * exp(2πi·j·k/n)|^2
        let theta = TAU * k as f32 / n as f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &val) in bspline_ints.iter().enumerate() {
            let phase = theta * j as f32;
            re += val * phase.cos();
            im += val * phase.sin();
        }
        let b2 = (re * re + im * im).max(1e-12);
        v[k] = 1.0 / b2; // store 1/|B|^2
    }
    v
}

/// Cubic B-spline weights for 4 neighbors; returns starting index and 4 weights.
/// Input s is in grid units (0..n), arbitrary real; we wrap indices to the grid.
fn bspline4_weights(s: f32) -> (isize, [f32; SPLINE_ORDER]) {
    let sfloor = s.floor();
    let u = s - sfloor; // fractional part in [0,1)
    let i0 = sfloor as isize - 1; // left-most point of 4-support

    // Cardinal cubic B-spline weights (order 4)
    let u2 = u * u;
    // let u3 = u2 * u;
    let u3 = u2.mul_add(u, 0.0);

    let w0 = (1.0 - u).powi(3) / 6.0;
    let w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    let w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    let w3 = u3 / 6.0;

    (i0, [w0, w1, w2, w3])
}

// experimenting
/// Returns (index, weights, derivative_weights)
fn bspline4_weights_and_derivs(s: f32) -> (isize, [f32; 4], [f32; 4]) {
    let sfloor = s.floor();
    let u = s - sfloor;
    let i0 = sfloor as isize - 1;

    let u2 = u * u;
    let u3 = u2 * u;

    // Standard Weights
    let w0 = (1.0 - u).powi(3) / 6.0;
    let w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    let w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    let w3 = u3 / 6.0;

    // Derivatives w.r.t u (d/du)
    // w0' = -3(1-u)^2 / 6 = -0.5 * (1-u)^2
    let dw0 = -0.5 * (1.0 - u).powi(2);
    // w1' = (9u^2 - 12u) / 6 = 1.5u^2 - 2u
    let dw1 = 1.5 * u2 - 2.0 * u;
    // w2' = (-9u^2 + 6u + 3) / 6 = -1.5u^2 + u + 0.5
    let dw2 = -1.5 * u2 + u + 0.5;
    // w3' = 3u^2 / 6 = 0.5 u^2
    let dw3 = 0.5 * u2;

    (i0, [w0, w1, w2, w3], [dw0, dw1, dw2, dw3])
}

fn wrap(i: isize, n: usize) -> usize {
    let n_isize = n as isize;
    let mut v = i % n_isize;
    if v < 0 {
        v += n_isize;
    }
    v as usize
}

/// For flexible molecules, computes the correction term.
/// May be useful for scaling corrections, e.g. bonded scaling and exlusions?
/// todo: This may not be suitable for general use.
pub fn force_correction(dir: Vec3, r: f32, qi: f32, qj: f32, alpha: f32) -> Vec3 {
    // Complement of the real-space Ewald kernel; this is what “belongs” to reciprocal.
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let ar = alpha * r;
    let fmag = qfac
        * (erf(ar as f64) as f32 * inv_r2 - (alpha * TWO_INV_SQRT_PI) * (-ar * ar).exp() * inv_r);
    // let fmag = qfac * (mul_add(erf(ar as f64) as f32, inv_r2,  -(alpha * TWO_INV_SQRT_PI)) * (-ar * ar).exp() * inv_r);
    dir * fmag
}

/// Used on both CPU and GPU paths
fn self_energy(q: &[f32], alpha: f32) -> f64 {
    -(alpha / SQRT_PI) as f64 * q.iter().map(|&qi| (qi as f64) * (qi as f64)).sum::<f64>()
}

fn gather_forces_from_potential(
    posits: &[Vec3],
    q: &[f32],
    phi_grid: &[f32], // This is the scalar potential grid (real space)
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
) -> Vec<Vec3> {
    let (nx, ny, nz) = plan_dims;
    let (lx, ly, lz) = box_dims;

    // Chain rule scaling factors: du/dx = N/L
    let fx_scl = nx as f32 / lx;
    let fy_scl = ny as f32 / ly;
    let fz_scl = nz as f32 / lz;

    posits
        .par_iter()
        .enumerate()
        .map(|(i, &r)| {
            let (ix0, wx, dwx) = bspline4_weights_and_derivs(r.x * fx_scl);
            let (iy0, wy, dwy) = bspline4_weights_and_derivs(r.y * fy_scl);
            let (iz0, wz, dwz) = bspline4_weights_and_derivs(r.z * fz_scl);

            let mut f_x = 0.0f64;
            let mut f_y = 0.0f64;
            let mut f_z = 0.0f64;

            for a in 0..4 {
                let ix = wrap(ix0 + a as isize, nx);

                for b in 0..4 {
                    let iy = wrap(iy0 + b as isize, ny);

                    // Precompute combined weights for outer loops
                    let w_xy = wx[a] * wy[b];
                    let dw_xy = dwx[a] * wy[b]; // for Fx
                    let w_dxy = wx[a] * dwy[b]; // for Fy

                    // Base index for Z-column
                    let base_idx = ix * (ny * nz) + iy * nz;

                    for c in 0..4 {
                        let iz = wrap(iz0 + c as isize, nz);
                        let idx = base_idx + iz;

                        let potential = phi_grid[idx] as f64;

                        // Fx component: phi * dwx * wy * wz
                        f_x += potential * (dw_xy * wz[c]) as f64;

                        // Fy component: phi * wx * dwy * wz
                        f_y += potential * (w_dxy * wz[c]) as f64;

                        // Fz component: phi * wx * wy * dwz
                        f_z += potential * (w_xy * dwz[c]) as f64;
                    }
                }
            }

            let qi = q[i] as f64;

            // F = -q * Gradient.
            // Gradient = (sum) * (N/L).
            Vec3 {
                x: (-f_x * qi * fx_scl as f64) as f32,
                y: (-f_y * qi * fy_scl as f64) as f32,
                z: (-f_z * qi * fz_scl as f64) as f32,
            }
        })
        .collect()
}

/// A utility fn. todo: QC this.
fn next_planner_n(mut n: usize) -> usize {
    fn good(mut x: usize) -> bool {
        for p in [2, 3, 5, 7] {
            while x.is_multiple_of(p) {
                x /= p;
            }
        }
        x == 1
    }
    if n < 2 {
        n = 2;
    }
    while !good(n) {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the B-spline modulus values for order-4 splines at the two analytic
    /// reference points.  `spline_bmod2_1d` stores 1/|B(k)|², so:
    ///   k = 0:   Σ M_4(j) = 1/6 + 4/6 + 1/6 = 1 → |B(0)|² = 1 → bmod2[0] = 1
    ///   k = N/2: Σ M_4(j)·e^{iπj} = 1/6 − 4/6 + 1/6 = −1/3 → |B(N/2)|² = 1/9 → bmod2[N/2] = 9
    #[test]
    fn test_bspline_bmod2_order4_known_values() {
        let n = 16;
        let v = spline_bmod_sq_inv_1d(n, 4);

        assert!(
            (v[0] - 1.0).abs() < 1e-5,
            "bmod2[0] = {}, expected 1.0",
            v[0]
        );
        assert!(
            (v[n / 2] - 9.0).abs() < 1e-3,
            "bmod2[N/2] = {}, expected 9.0",
            v[n / 2]
        );
    }

    /// All 1/|B|² values must be finite and positive (no degenerate modes blow up).
    #[test]
    fn test_bspline_bmod2_all_finite_positive() {
        for n in [8usize, 16, 32, 50] {
            let v = spline_bmod_sq_inv_1d(n, 4);
            for (k, &val) in v.iter().enumerate() {
                assert!(
                    val.is_finite() && val > 0.0,
                    "bmod2[{k}] = {val} is not finite/positive (n={n})"
                );
            }
        }
    }
}

/// A utility function to get the (nx, ny, nz) tuple of plan dimensions based
/// on grid dimensions, and mesh spacing. A mesh spacing of 1Å is a good starting point.
/// Pass this into the `PmeRecip::new()` constructor, or set these values
/// up with some other approach.
pub fn get_grid_n(l: (f32, f32, f32), mesh_spacing: f32) -> (usize, usize, usize) {
    let (lx, ly, lz) = l;

    let nx0 = (lx / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;
    let ny0 = (ly / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;
    let nz0 = (lz / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;

    let nx = next_planner_n(nx0);
    let ny = next_planner_n(ny0);
    let mut nz = next_planner_n(nz0);

    // We use this because we use Z as the fast (contiguous) axis.
    if !nz.is_multiple_of(2) {
        nz = next_planner_n(nz + 1);
    }

    (nx, ny, nz)
}
