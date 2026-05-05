// Shared utilities for the example programs: a tiny RNG, lognormal Cox field
// generators in 2D and 3D, and a Poisson point generator. These are not part of
// the morton_cascade library because they are test-data scaffolding, not
// algorithms users would call from their own code.
//
// Included via #[path = "common.rs"] mod common; in each example.

#![allow(dead_code)]

// ---------- Tiny xorshift PRNG (deterministic, no deps) ----------

pub struct Rng { s: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Self { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) } }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.s = x; x
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 { return 0; }
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k: u32 = 0;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.uniform();
                if p <= l { return k - 1; }
            }
        } else {
            let g = lambda + lambda.sqrt() * self.normal();
            g.round().max(0.0) as u32
        }
    }
}

// ---------- Uniform Poisson process ----------

pub fn gen_poisson_process(n_target: usize, rng: &mut Rng) -> Vec<(u16, u16)> {
    let scale = (1u32 << 16) as f64;
    let mut pts = Vec::with_capacity(n_target);
    let n_actual = rng.poisson(n_target as f64) as usize;
    for _ in 0..n_actual {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        pts.push((x, y));
    }
    pts
}

pub fn gen_poisson_process_3d(n_target: usize, rng: &mut Rng) -> Vec<(u16, u16, u16)> {
    let scale = (1u32 << 16) as f64;
    let mut pts = Vec::with_capacity(n_target);
    let n_actual = rng.poisson(n_target as f64) as usize;
    for _ in 0..n_actual {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let z = (rng.uniform() * scale).min(scale - 1.0) as u16;
        pts.push((x, y, z));
    }
    pts
}

// ---------- 2D lognormal Cox field ----------

pub struct LogNormalField {
    pub g: usize,
    pub sigma2_g: f64,
    pub field: Vec<f64>,
}

impl LogNormalField {
    pub fn new(g: usize, alpha: f64, target_sigma2: f64, n_modes: usize, rng: &mut Rng) -> Self {
        let k_min = 2.0 * std::f64::consts::PI / (g as f64);
        let k_max = std::f64::consts::PI;
        let mut amps = Vec::with_capacity(n_modes);
        let mut ks   = Vec::with_capacity(n_modes);
        let mut phis = Vec::with_capacity(n_modes);
        let mut weights = Vec::with_capacity(n_modes);
        let mut wsum = 0.0;
        for _ in 0..n_modes {
            let u = rng.uniform();
            let k = (k_min.ln() * (1.0 - u) + k_max.ln() * u).exp();
            let theta = 2.0 * std::f64::consts::PI * rng.uniform();
            let kx = k * theta.cos();
            let ky = k * theta.sin();
            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
            let w = k.powf(-alpha);
            wsum += w;
            ks.push((kx, ky));
            phis.push(phi);
            weights.push(w);
        }
        for &w in &weights {
            let var_mode = target_sigma2 * w / wsum;
            amps.push((2.0 * var_mode).sqrt());
        }

        let mut field = vec![0.0f64; g*g];
        let mut s = 0.0; let mut s2 = 0.0;
        for iy in 0..g {
            for ix in 0..g {
                let mut v = 0.0;
                for m in 0..n_modes {
                    let (kx, ky) = ks[m];
                    v += amps[m] * (kx * ix as f64 + ky * iy as f64 + phis[m]).cos();
                }
                field[iy*g + ix] = v;
                s += v; s2 += v*v;
            }
        }
        let n = (g*g) as f64;
        let mean = s/n; let var = s2/n - mean*mean;
        for v in field.iter_mut() { *v -= mean; }
        Self { g, sigma2_g: var, field }
    }

    pub fn lambda(&self, n_target: usize) -> Vec<f64> {
        let mut lam = vec![0.0f64; self.g * self.g];
        let mut tot = 0.0;
        for i in 0..self.g*self.g {
            let l = (self.field[i] - 0.5*self.sigma2_g).exp();
            lam[i] = l; tot += l;
        }
        let scale = n_target as f64 / tot;
        for v in lam.iter_mut() { *v *= scale; }
        lam
    }

    pub fn sample(&self, n_target: usize, rng: &mut Rng) -> Vec<(u16, u16)> {
        let g = self.g;
        let lam = self.lambda(n_target);
        let scale_xy = (1u32 << 16) as f64;
        let cell_w = scale_xy / g as f64;
        let mut pts = Vec::with_capacity(n_target);
        for iy in 0..g {
            for ix in 0..g {
                let n_cell = rng.poisson(lam[iy*g + ix]);
                for _ in 0..n_cell {
                    let fx = (ix as f64 + rng.uniform()) * cell_w;
                    let fy = (iy as f64 + rng.uniform()) * cell_w;
                    let x = fx.min(scale_xy - 1.0) as u16;
                    let y = fy.min(scale_xy - 1.0) as u16;
                    pts.push((x, y));
                }
            }
        }
        pts
    }
}

// ---------- 3D lognormal Cox field ----------

pub struct LogNormalField3D {
    pub g: usize,
    pub sigma2_g: f64,
    pub field: Vec<f64>,
}

impl LogNormalField3D {
    pub fn new(g: usize, alpha: f64, target_sigma2: f64, n_modes: usize, rng: &mut Rng) -> Self {
        let k_min = 2.0 * std::f64::consts::PI / (g as f64);
        let k_max = std::f64::consts::PI;
        let mut amps = Vec::with_capacity(n_modes);
        let mut ks   = Vec::with_capacity(n_modes);
        let mut phis = Vec::with_capacity(n_modes);
        let mut weights = Vec::with_capacity(n_modes);
        let mut wsum = 0.0;
        for _ in 0..n_modes {
            let u = rng.uniform();
            let k = (k_min.ln() * (1.0 - u) + k_max.ln() * u).exp();
            let cos_theta = 2.0 * rng.uniform() - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi_angle = 2.0 * std::f64::consts::PI * rng.uniform();
            let kx = k * sin_theta * phi_angle.cos();
            let ky = k * sin_theta * phi_angle.sin();
            let kz = k * cos_theta;
            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
            let w = k.powf(-alpha);
            wsum += w;
            ks.push((kx, ky, kz));
            phis.push(phi);
            weights.push(w);
        }
        for &w in &weights {
            let var_mode = target_sigma2 * w / wsum;
            amps.push((2.0 * var_mode).sqrt());
        }

        let mut field = vec![0.0f64; g*g*g];
        let mut s = 0.0; let mut s2 = 0.0;
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let mut v = 0.0;
                    for m in 0..n_modes {
                        let (kx, ky, kz) = ks[m];
                        v += amps[m] * (kx * ix as f64 + ky * iy as f64 + kz * iz as f64 + phis[m]).cos();
                    }
                    field[(iz*g + iy)*g + ix] = v;
                    s += v; s2 += v*v;
                }
            }
        }
        let n = (g*g*g) as f64;
        let mean = s/n; let var = s2/n - mean*mean;
        for v in field.iter_mut() { *v -= mean; }
        Self { g, sigma2_g: var, field }
    }

    pub fn lambda(&self, n_target: usize) -> Vec<f64> {
        let mut lam = vec![0.0f64; self.g.pow(3)];
        let mut tot = 0.0;
        for i in 0..self.g.pow(3) {
            let l = (self.field[i] - 0.5*self.sigma2_g).exp();
            lam[i] = l; tot += l;
        }
        let scale = n_target as f64 / tot;
        for v in lam.iter_mut() { *v *= scale; }
        lam
    }

    pub fn sample(&self, n_target: usize, rng: &mut Rng) -> Vec<(u16, u16, u16)> {
        let g = self.g;
        let lam = self.lambda(n_target);
        let scale_xy = (1u32 << 16) as f64;
        let cell_w = scale_xy / g as f64;
        let mut pts = Vec::with_capacity(n_target);
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let n_cell = rng.poisson(lam[(iz*g + iy)*g + ix]);
                    for _ in 0..n_cell {
                        let fx = (ix as f64 + rng.uniform()) * cell_w;
                        let fy = (iy as f64 + rng.uniform()) * cell_w;
                        let fz = (iz as f64 + rng.uniform()) * cell_w;
                        let x = fx.min(scale_xy - 1.0) as u16;
                        let y = fy.min(scale_xy - 1.0) as u16;
                        let z = fz.min(scale_xy - 1.0) as u16;
                        pts.push((x, y, z));
                    }
                }
            }
        }
        pts
    }
}
