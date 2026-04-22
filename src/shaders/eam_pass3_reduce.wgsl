// =============================================================================
// eam_pass3_reduce.wgsl  —  workgroup tree reduction for energy summation
//
// Stand-alone shader: common.wgsl is NOT prepended.
// Each workgroup (64 threads) reads up to 64 consecutive values from `input`,
// performs a tree reduction, and writes one partial sum to `output`.
//
// Dispatch pattern (engine.rs drives the loop):
//   Pass 3a: dispatch ceil(N/64) workgroups  →  partial_sums[ceil(N/64)]
//   Pass 3b: dispatch 1 workgroup            →  energy_total[1]
//            (correct for N ≤ 4096; engine falls back to CPU sum for larger N)
//
// Bindings (no common.wgsl):
//   binding 0 — input:          array<f32>  (read)
//   binding 1 — output:         array<f32>  (read_write)
//   binding 2 — reduce_params:  ReduceParams uniform
// =============================================================================

struct ReduceParams {
    count: u32,   // number of valid input elements this dispatch
    _p0:   u32,
    _p1:   u32,
    _p2:   u32,
}

@group(0) @binding(0) var<storage, read>       input:         array<f32>;
@group(0) @binding(1) var<storage, read_write> output:        array<f32>;
@group(0) @binding(2) var<uniform>             reduce_params: ReduceParams;

var<workgroup> wg_data: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_index) lid:  u32,
    @builtin(workgroup_id)           wgid: vec3<u32>,
) {
    let count = reduce_params.count;

    // Load: threads beyond `count` contribute 0 (identity for addition).
    //
    // Use an explicit if-guard rather than `select(0.0, input[gid.x], cond)`.
    // In WGSL, select() evaluates both operands eagerly before choosing one,
    // so `input[gid.x]` would be read even when `gid.x >= count`, which is
    // an out-of-bounds access (the input buffer has exactly `count` elements).
    // WebGPU's robustness rules return 0 for OOB reads, but native backends
    // without robustness exhibit undefined behaviour.  The if-guard prevents
    // the read entirely.
    var v: f32 = 0.0;
    if gid.x < count { v = input[gid.x]; }
    wg_data[lid] = v;
    workgroupBarrier();

    // Binary tree reduction within workgroup.
    // 6 levels for workgroup_size = 64.
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            wg_data[lid] += wg_data[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the partial sum for this workgroup.
    if lid == 0u {
        output[wgid.x] = wg_data[0];
    }
}
