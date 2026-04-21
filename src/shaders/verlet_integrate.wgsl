// =============================================================================
// verlet_integrate.wgsl  —  leapfrog Velocity Verlet integration
//
// Stand-alone shader (common.wgsl is NOT prepended).
//
// Algorithm (leapfrog variant):
//   a(t)      = F(t) / m                      [F in eV/Å, m in amu]
//   v(t+½dt)  = v(t-½dt) + a(t) * dt
//   r(t+dt)   = r(t)     + v(t+½dt) * dt
//
// Unit system: eV / Å / ps
//   inv_mass  = 1 / (m_amu × 9648.5)  [Å·ps² / (eV·ps²) = 1/amu in these units]
//
// Zero-division guard:
//   If inv_mass == 0.0 (caller mistake), acceleration is zero — positions and
//   velocities are unchanged. This is physically wrong but avoids NaN/Inf.
//
// Bindings:
//   0  positions   array<vec4<f32>>  read_write  (xyz used; w=0)
//   1  velocities  array<vec4<f32>>  read_write  (xyz used; w=0)
//   2  forces      array<vec4<f32>>  read        (xyz = force in eV/Å)
//   3  vp          VerletParams      uniform
// =============================================================================

struct VerletParams {
    n_atoms:  u32,
    dt:       f32,       // timestep [ps]
    inv_mass: f32,       // 1 / (m_amu * 9648.5)
    _pad:     u32,
}

@group(0) @binding(0) var<storage, read_write> positions:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       forces:     array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             vp:         VerletParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= vp.n_atoms { return; }

    // Guard: if inv_mass is zero, skip integration (acceleration = 0).
    // This prevents NaN propagation when the caller passes inv_mass = 0.
    if vp.inv_mass == 0.0 { return; }

    // a = F / m  (eV/Å × Å·ps²/(eV) = Å/ps²)
    let a = forces[i].xyz * vp.inv_mass;

    // Leapfrog half-step velocity, then full-step position.
    let v_half = velocities[i].xyz + a * vp.dt;
    let r_new  = positions[i].xyz  + v_half * vp.dt;

    velocities[i] = vec4<f32>(v_half, 0.0);
    positions[i]  = vec4<f32>(r_new,  0.0);
}
