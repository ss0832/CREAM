// =============================================================================
// eam_pass3_energy.wgsl  —  Pass 3: energy pass-through stub
//
// Stand-alone shader (common.wgsl is NOT prepended — it only needs the
// energy buffer).
//
// Strategy: per-atom energies live in energy_per_atom[0..N].  The CPU reads
// them back and sums them to get the total.  This shader loads each value
// into workgroup shared memory (confirming the dispatch pipeline works) and
// writes it back unchanged.  If CPU-sum precision ever becomes inadequate
// (|error| > 1e-5 eV), replace this stub with a full workgroup tree-reduction
// (writes partial sums to energy_per_atom[workgroup_id], then a second
// dispatch sums those).
// =============================================================================

@group(0) @binding(0) var<storage, read_write> energy_per_atom: array<f32>;

var<workgroup> wg_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n = arrayLength(&energy_per_atom);
    if gid.x < n {
        wg_sum[lid] = energy_per_atom[gid.x];
    } else {
        wg_sum[lid] = 0.0;
    }
    workgroupBarrier();
    // Passthrough: write unchanged (CPU sums all N values after readback).
    if gid.x < n {
        energy_per_atom[gid.x] = wg_sum[lid];
    }
}
