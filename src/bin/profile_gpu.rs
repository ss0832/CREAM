//! GPU per-pass timing breakdown using wgpu TIMESTAMP_QUERY.
//!
//! Each compute pass is surrounded by `encoder.write_timestamp()` calls.
//! On a real GPU these measure true shader execution time; on llvmpipe
//! (software Vulkan) they measure CPU-thread execution, which is equally
//! accurate because `device.poll(Maintain::Wait)` is a hard synchronisation
//! point in both cases.
//!
//! Run with the Mesa software Vulkan driver (no physical GPU required):
//!
//!   XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan \
//!     cargo run --release --bin profile_gpu 2>/dev/null
//!
//! On a physical GPU (Vulkan or Metal):
//!   cargo run --release --bin profile_gpu 2>/dev/null
//!
//! Timestamp slots used per frame
//! ─────────────────────────────────────────────────────────────────
//!  CellList profile  (two submissions — CPU prefix-sum in between)
//!
//!  Submission A  (pass 0a + 0b + count readback)
//!    T[0]  before pass0a  (Morton ID assignment)
//!    T[1]  after  pass0a
//!    T[2]  before pass0b  (atomicAdd cell counts)
//!    T[3]  after  pass0b  + copy_counts_to_rb submitted
//!
//!  [CPU phase]: map rb_counts, prefix-sum, upload cell_start & write_offsets
//!
//!  Submission B  (pass 0c + 1 + 2 + 3a + 3b + readbacks)
//!    T[4]  before pass0c  (scatter sorted_atoms)
//!    T[5]  after  pass0c
//!    T[6]  before pass1   (density accumulation)
//!    T[7]  after  pass1
//!    T[8]  before pass2   (forces + per-atom energy)
//!    T[9]  after  pass2
//!    T[10] before pass3a  (reduce: N atoms → ceil(N/64) partials)
//!    T[11] after  pass3a
//!    T[12] before pass3b  (reduce: ceil(N/64) partials → 1)
//!    T[13] after  pass3b
//!
//!  AllPairs profile  (single submission — no CPU round-trip)
//!    T[0]  before pass1
//!    T[1]  after  pass1
//!    T[2]  before pass2
//!    T[3]  after  pass2
//!    T[4]  before pass3a
//!    T[5]  after  pass3a
//!    T[6]  before pass3b
//!    T[7]  after  pass3b
//! ─────────────────────────────────────────────────────────────────
//!
//! NOTE (v0.1.0-beta.10+): this binary predates pass0d (Morton-order
//! reorder) and the pass1/pass2 CellList debug-flags binding (slot 10).
//! It remains in the tree as a reference for the older pipeline layout
//! but is gated behind the `profile-gpu-bin` Cargo feature so a default
//! `cargo build` does not attempt to compile an out-of-date bind-group
//! schema against the current shaders.  To run it, check out a
//! pre-beta.10 tag or rebuild with that feature after updating it.

#![cfg(feature = "profile-gpu-bin")]
#![allow(dead_code, unused_variables, unused_imports)]

use bytemuck::{Pod, Zeroable};
use cream::potential::{eam::EamPotential, GpuPotential, PotentialGpuBuffers};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use wgpu::util::DeviceExt;

// ── Knobs ────────────────────────────────────────────────────────────────────
const CELL_SIZE: f32 = 5.5;
const WARMUP: usize = 5;
const MEASURE: usize = 20;

// Atom counts to profile: N = 256, 500, 864, 2048, 4000
const REPS: &[usize] = &[4, 5, 6, 8, 10];

// ── GPU uniform structs (must match engine.rs / common.wgsl exactly) ─────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimParams {
    n_atoms: u32,
    n_elem: u32,
    cutoff_sq: f32,
    min_dist_sq: f32,
    h0: [f32; 4],
    h1: [f32; 4],
    h2: [f32; 4],
    hinv_col0: [f32; 4],
    hinv_col1: [f32; 4],
    hinv_col2: [f32; 4],
    use_cell_list: u32,
    use_pbc: u32,
    _pad0: u32,
    _pad1: u32,
    n_cells_x: u32,
    n_cells_y: u32,
    n_cells_z: u32,
    n_cells_total: u32,
    cell_size: f32,
    n_cells_x_pad: u32,
    n_cells_y_pad: u32,
    n_cells_z_pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// ── Shader sources ───────────────────────────────────────────────────────────
const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
const PASS0A_WGSL: &str = include_str!("../shaders/cell_pass0a_assign.wgsl");
const PASS0B_WGSL: &str = include_str!("../shaders/cell_pass0b_sort.wgsl");
const PASS0C_WGSL: &str = include_str!("../shaders/cell_pass0c_range.wgsl");
const REDUCE_WGSL: &str = include_str!("../shaders/eam_pass3_reduce.wgsl");

const WG: u32 = 64;

// ── Helpers ───────────────────────────────────────────────────────────────────
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:6.0} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:6.1} µs", ns / 1_000.0)
    } else {
        format!("{:6.2} ms", ns / 1_000_000.0)
    }
}

fn alloc_storage(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn alloc_uniform(device: &wgpu::Device, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (size as u64).max(16),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn alloc_readback(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn make_pipeline(
    device: &wgpu::Device,
    src: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn n_cells_from_box(l: f32, cs: f32) -> u32 {
    ((l / cs).floor() as u32).max(1)
}

/// Blocking map-read for a small buffer; returns raw bytes.
fn map_read_sync<'a>(device: &wgpu::Device, buf: &'a wgpu::Buffer) -> wgpu::BufferView<'a> {
    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |_| {
        done2.store(true, Ordering::Release);
    });
    device.poll(wgpu::Maintain::Wait);
    assert!(done.load(Ordering::Acquire));
    buf.slice(..).get_mapped_range()
}

/// Convert raw u64 timestamp array + period into nanoseconds per delta.
fn ts_delta_ns(raw: &[u64], a: usize, b: usize, period_ns: f32) -> f64 {
    (raw[b].saturating_sub(raw[a])) as f64 * period_ns as f64
}

// ── FCC supercell builder ─────────────────────────────────────────────────────
fn fcc_supercell(rep: usize) -> (Vec<[f32; 4]>, Vec<u32>, f32) {
    let a = 3.615_f32;
    let basis: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [a / 2., a / 2., 0.0],
        [a / 2., 0.0, a / 2.],
        [0.0, a / 2., a / 2.],
    ];
    let n = rep * rep * rep * 4;
    let mut pos = Vec::with_capacity(n);
    let mut types = Vec::with_capacity(n);
    for ix in 0..rep {
        for iy in 0..rep {
            for iz in 0..rep {
                for b in &basis {
                    pos.push([
                        b[0] + ix as f32 * a,
                        b[1] + iy as f32 * a,
                        b[2] + iz as f32 * a,
                        0.0,
                    ]);
                    types.push(0u32);
                }
            }
        }
    }
    (pos, types, a * rep as f32)
}

// ── Synthetic Cu EAM potential (same as bench_gpu) ───────────────────────────
fn synth_pot() -> EamPotential {
    use std::fmt::Write;
    let (nr, nrho) = (200u32, 200u32);
    let (dr, drho, cutoff) = (0.025_f32, 0.01_f32, 4.95_f32);
    let mut s = String::new();
    writeln!(s, "# Synthetic Cu\n# bench\n# ok\n1 Cu").unwrap();
    writeln!(s, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    writeln!(s, "29 63.546 3.615 fcc").unwrap();
    for i in 0..nrho {
        let rho = i as f32 * drho;
        write!(s, "{:.8e} ", -(rho + 0.01_f32).sqrt()).unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r < cutoff {
            let t = cutoff - r;
            t * t / (cutoff * cutoff)
        } else {
            0.0
        };
        write!(s, "{v:.8e} ").unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r > 0.5 && r < cutoff {
            let t = cutoff - r;
            t * t * t / (r * cutoff * cutoff * cutoff)
        } else {
            0.0
        };
        write!(s, "{v:.8e} ").unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    EamPotential::from_str(&s).expect("synth_pot failed")
}

// ── Per-pass timing result ────────────────────────────────────────────────────
#[derive(Default, Clone)]
struct PassTimings {
    pass0a_ns: f64,       // Morton ID assign
    pass0b_ns: f64,       // atomicAdd counts
    cpu_prefix_ns: f64,   // CPU prefix sum + upload (Instant)
    pass0c_ns: f64,       // scatter sorted_atoms
    pass1_ns: f64,        // density
    pass2_ns: f64,        // forces + energy
    pass3a_ns: f64,       // reduce atoms→partials
    pass3b_ns: f64,       // reduce partials→total
    readback_cpu_ns: f64, // readback copy + map (Instant)
}

#[allow(dead_code)]
impl PassTimings {
    fn cell_list_gpu_ns(&self) -> f64 {
        self.pass0a_ns
            + self.pass0b_ns
            + self.pass0c_ns
            + self.pass1_ns
            + self.pass2_ns
            + self.pass3a_ns
            + self.pass3b_ns
    }
    fn total_ns(&self) -> f64 {
        self.cell_list_gpu_ns() + self.cpu_prefix_ns + self.readback_cpu_ns
    }
}

// ── Core profiler ─────────────────────────────────────────────────────────────
struct Profiler {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pot: EamPotential,
    ts_period: f32,
    has_ts: bool,
}

impl Profiler {
    async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("No GPU adapter found — try WGPU_BACKEND=vulkan");

        let want =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let has_ts = adapter.features().contains(want);

        let features = if has_ts {
            want
        } else {
            wgpu::Features::empty()
        };
        if !has_ts {
            eprintln!("⚠  TIMESTAMP_QUERY not supported; falling back to Instant timing");
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: features,
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 12,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Device creation failed");

        let ts_period = queue.get_timestamp_period(); // ns per tick

        let info = adapter.get_info();
        println!("Adapter : {} ({:?})", info.name, info.device_type);
        println!("Backend : {:?}", info.backend);
        println!("TS period: {:.2} ns/tick  has_ts={has_ts}", ts_period);

        Self {
            device,
            queue,
            pot: synth_pot(),
            ts_period,
            has_ts,
        }
    }

    // ── Profile CellList at a given atom count ────────────────────────────────
    fn profile_cellist(&self, rep: usize, n_iter: usize) -> Vec<PassTimings> {
        let (pos, types, box_l) = fcc_supercell(rep);
        let n = pos.len();
        let h = [[box_l, 0., 0.], [0., box_l, 0.], [0., 0., box_l]];
        let hinv = 1.0 / box_l;
        let rc = self.pot.cutoff();
        let layout = self.pot.buffer_layout();
        let nc = n_cells_from_box(box_l, CELL_SIZE);
        let nc_pad = nc.next_power_of_two();
        let nct = nc * nc * nc;
        let n_morton = (nc_pad * nc_pad * nc_pad) as usize;
        let num_partial = n.div_ceil(64).max(1);

        // ── Potential tables ─────────────────────────────────────────────────
        let PotentialGpuBuffers {
            tables_buf,
            layout_buf,
        } = self.pot.upload_tables(&self.device, &self.queue).unwrap();

        // ── Frame buffers ─────────────────────────────────────────────────────
        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&pos),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let types_buf = alloc_storage(&self.device, 4 * n as u64);
        let params_buf = alloc_uniform(&self.device, std::mem::size_of::<GpuSimParams>());
        let density_buf = alloc_storage(&self.device, 4 * n as u64);
        let forces_buf = alloc_storage(&self.device, layout.output_stride_bytes * n as u64);
        let epa_buf = alloc_storage(&self.device, 4 * n as u64);
        let partial_buf = alloc_storage(&self.device, 4 * num_partial as u64);
        let energy_buf = alloc_storage(&self.device, 4);
        let rp_buf = alloc_uniform(&self.device, 16);
        let rb_forces = alloc_readback(&self.device, layout.output_stride_bytes * n as u64);
        let rb_epa = alloc_readback(&self.device, 4 * n as u64);
        let rb_etot = alloc_readback(&self.device, 4);

        // Cell list buffers
        let cell_ids_buf = alloc_storage(&self.device, 4 * n as u64);
        let sorted_buf = alloc_storage(&self.device, 4 * n as u64);
        let cell_start_buf = alloc_storage(&self.device, 4 * (n_morton + 1) as u64);
        let cell_counts_buf = alloc_storage(&self.device, 4 * n_morton as u64);
        let write_off_buf = alloc_storage(&self.device, 4 * n_morton as u64);
        let rb_counts = alloc_readback(&self.device, 4 * n_morton as u64);

        // Timestamp query set (14 slots for CellList)
        const N_TS: u32 = 14;
        let ts_qs = self.has_ts.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile_ts"),
                ty: wgpu::QueryType::Timestamp,
                count: N_TS,
            })
        });
        let ts_resolve_buf = self.has_ts.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8 * N_TS as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let ts_rb = self
            .has_ts
            .then(|| alloc_readback(&self.device, 8 * N_TS as u64));

        // ── Build SimParams ───────────────────────────────────────────────────
        let params = GpuSimParams {
            n_atoms: n as u32,
            n_elem: 1,
            cutoff_sq: rc * rc,
            min_dist_sq: 1e-4,
            h0: [h[0][0], h[0][1], h[0][2], 0.],
            h1: [h[1][0], h[1][1], h[1][2], 0.],
            h2: [h[2][0], h[2][1], h[2][2], 0.],
            hinv_col0: [hinv, 0., 0., 0.],
            hinv_col1: [0., hinv, 0., 0.],
            hinv_col2: [0., 0., hinv, 0.],
            use_cell_list: 1,
            use_pbc: 1,
            _pad0: 0,
            _pad1: 0,
            n_cells_x: nc,
            n_cells_y: nc,
            n_cells_z: nc,
            n_cells_total: nct,
            cell_size: CELL_SIZE,
            n_cells_x_pad: nc_pad,
            n_cells_y_pad: nc_pad,
            n_cells_z_pad: nc_pad,
        };
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&types_buf, 0, bytemuck::cast_slice(&types));

        // ── Build pipelines (once, reused across iterations) ──────────────────
        macro_rules! bgl {
            ($entries:expr) => {
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: $entries,
                    })
            };
        }
        macro_rules! bg {
            ($bgl:expr, $slots:expr) => {{
                let entries: Vec<wgpu::BindGroupEntry<'_>> = $slots
                    .into_iter()
                    .map(|(b, buf): (u32, &wgpu::Buffer)| wgpu::BindGroupEntry {
                        binding: b,
                        resource: buf.as_entire_binding(),
                    })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &$bgl,
                    entries: &entries,
                })
            }};
        }

        use wgpu::{BindingType, BufferBindingType as BBT};
        let stor_r = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BBT::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let stor_rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BBT::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BBT::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let src0a = format!("{COMMON_WGSL}\n{PASS0A_WGSL}");
        let src0b = format!("{COMMON_WGSL}\n{PASS0B_WGSL}");
        let src0c = format!("{COMMON_WGSL}\n{PASS0C_WGSL}");
        let src1 = self.pot.pass1_cellist_shader().unwrap().into_owned();
        let src2 = self.pot.pass2_cellist_shader().unwrap().into_owned();

        let bgl0a = bgl!(&[stor_r(0), uni(2), stor_rw(5)]);
        let bgl0b = bgl!(&[stor_r(0), uni(2), stor_r(5), stor_rw(6)]);
        let bgl0c = bgl!(&[stor_r(0), uni(2), stor_r(5), stor_rw(6), stor_rw(7)]);
        let bgl1 = bgl!(&[
            stor_r(0),
            stor_r(1),
            uni(2),
            stor_r(3),
            uni(4),
            stor_rw(5),
            stor_r(6),
            stor_r(7)
        ]);
        let bgl2 = bgl!(&[
            stor_r(0),
            stor_r(1),
            uni(2),
            stor_r(3),
            uni(4),
            stor_r(5),
            stor_rw(6),
            stor_rw(7),
            stor_r(8),
            stor_r(9)
        ]);
        let bgl3 = bgl!(&[stor_r(0), stor_rw(1), uni(2)]);

        let pl0a = make_pipeline(&self.device, &src0a, &bgl0a);
        let pl0b = make_pipeline(&self.device, &src0b, &bgl0b);
        let pl0c = make_pipeline(&self.device, &src0c, &bgl0c);
        let pl1 = make_pipeline(&self.device, &src1, &bgl1);
        let pl2 = make_pipeline(&self.device, &src2, &bgl2);
        let pl3 = make_pipeline(&self.device, REDUCE_WGSL, &bgl3);

        let bg0a = bg!(
            bgl0a,
            [(0u32, &pos_buf), (2, &params_buf), (5, &cell_ids_buf)]
        );
        let bg0b = bg!(
            bgl0b,
            [
                (0u32, &pos_buf),
                (2, &params_buf),
                (5, &cell_ids_buf),
                (6, &cell_counts_buf)
            ]
        );
        let bg0c = bg!(
            bgl0c,
            [
                (0u32, &pos_buf),
                (2, &params_buf),
                (5, &cell_ids_buf),
                (6, &write_off_buf),
                (7, &sorted_buf)
            ]
        );
        let bg1 = bg!(
            bgl1,
            [
                (0u32, &pos_buf),
                (1, &types_buf),
                (2, &params_buf),
                (3, &tables_buf),
                (4, &layout_buf),
                (5, &density_buf),
                (6, &sorted_buf),
                (7, &cell_start_buf)
            ]
        );
        let bg2 = bg!(
            bgl2,
            [
                (0u32, &pos_buf),
                (1, &types_buf),
                (2, &params_buf),
                (3, &tables_buf),
                (4, &layout_buf),
                (5, &density_buf),
                (6, &forces_buf),
                (7, &epa_buf),
                (8, &sorted_buf),
                (9, &cell_start_buf)
            ]
        );
        let bg3a = bg!(bgl3, [(0u32, &epa_buf), (1, &partial_buf), (2, &rp_buf)]);
        let bg3b = bg!(bgl3, [(0u32, &partial_buf), (1, &energy_buf), (2, &rp_buf)]);

        let dispatch = |enc: &mut wgpu::CommandEncoder,
                        pl: &wgpu::ComputePipeline,
                        bg: &wgpu::BindGroup,
                        nd: usize| {
            let groups = (nd as u32).div_ceil(WG).max(1);
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        };

        // ── Run iterations ────────────────────────────────────────────────────
        let mut results = Vec::with_capacity(n_iter);

        for _ in 0..n_iter {
            let mut t = PassTimings::default();

            // ─── Submission A: pass 0a + 0b ───────────────────────────────────
            {
                self.queue.write_buffer(
                    &cell_counts_buf,
                    0,
                    bytemuck::cast_slice(&vec![0u32; n_morton]),
                );
                let mut enc = self.device.create_command_encoder(&Default::default());

                if let (Some(qs), Some(rb)) = (ts_qs.as_ref(), ts_resolve_buf.as_ref()) {
                    enc.write_timestamp(qs, 0);
                    dispatch(&mut enc, &pl0a, &bg0a, n);
                    enc.write_timestamp(qs, 1);
                    enc.write_timestamp(qs, 2);
                    dispatch(&mut enc, &pl0b, &bg0b, n);
                    enc.write_timestamp(qs, 3);
                    // copy counts for CPU readback
                    enc.copy_buffer_to_buffer(
                        &cell_counts_buf,
                        0,
                        &rb_counts,
                        0,
                        4 * n_morton as u64,
                    );
                    // resolve timestamps 0..4 into resolve buffer
                    enc.resolve_query_set(qs, 0..4, rb, 0);
                    enc.copy_buffer_to_buffer(rb, 0, ts_rb.as_ref().unwrap(), 0, 8 * 4);
                } else {
                    dispatch(&mut enc, &pl0a, &bg0a, n);
                    dispatch(&mut enc, &pl0b, &bg0b, n);
                    enc.copy_buffer_to_buffer(
                        &cell_counts_buf,
                        0,
                        &rb_counts,
                        0,
                        4 * n_morton as u64,
                    );
                }
                self.queue.submit([enc.finish()]);
                self.device.poll(wgpu::Maintain::Wait);
            }

            // Read submission-A timestamps
            if self.has_ts {
                let raw_view = map_read_sync(&self.device, ts_rb.as_ref().unwrap());
                let raw: &[u64] = bytemuck::cast_slice(&raw_view);
                let p = self.ts_period;
                t.pass0a_ns = ts_delta_ns(raw, 0, 1, p);
                t.pass0b_ns = ts_delta_ns(raw, 2, 3, p);
                drop(raw_view);
                ts_rb.as_ref().unwrap().unmap();
            }

            // ─── CPU: prefix sum ───────────────────────────────────────────────
            let cpu0 = Instant::now();
            let cell_counts: Vec<u32> = {
                let view = map_read_sync(&self.device, &rb_counts);
                let v: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
                drop(view);
                rb_counts.unmap();
                v
            };
            let mut cell_start = vec![0u32; n_morton + 1];
            for m in 0..n_morton {
                cell_start[m + 1] = cell_start[m] + cell_counts[m];
            }
            self.queue
                .write_buffer(&cell_start_buf, 0, bytemuck::cast_slice(&cell_start));
            self.queue.write_buffer(
                &write_off_buf,
                0,
                bytemuck::cast_slice(&cell_start[..n_morton]),
            );
            t.cpu_prefix_ns = cpu0.elapsed().as_nanos() as f64;

            // ─── Submission B: pass 0c + 1 + 2 + 3a + 3b + readbacks ──────────
            {
                self.queue.write_buffer(
                    &rp_buf,
                    0,
                    bytemuck::bytes_of(&ReduceParams {
                        count: n as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    }),
                );
                let mut enc = self.device.create_command_encoder(&Default::default());

                if let (Some(qs), Some(rb)) = (ts_qs.as_ref(), ts_resolve_buf.as_ref()) {
                    enc.write_timestamp(qs, 4);
                    dispatch(&mut enc, &pl0c, &bg0c, n);
                    enc.write_timestamp(qs, 5);

                    enc.write_timestamp(qs, 6);
                    dispatch(&mut enc, &pl1, &bg1, n);
                    enc.write_timestamp(qs, 7);

                    enc.write_timestamp(qs, 8);
                    dispatch(&mut enc, &pl2, &bg2, n);
                    enc.write_timestamp(qs, 9);

                    // Pass 3a: reduce N atoms → ceil(N/64) partials
                    self.queue.write_buffer(
                        &rp_buf,
                        0,
                        bytemuck::bytes_of(&ReduceParams {
                            count: n as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        }),
                    );
                    enc.write_timestamp(qs, 10);
                    dispatch(&mut enc, &pl3, &bg3a, n);
                    enc.write_timestamp(qs, 11);

                    // Pass 3b: reduce partials → 1
                    self.queue.write_buffer(
                        &rp_buf,
                        0,
                        bytemuck::bytes_of(&ReduceParams {
                            count: num_partial as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        }),
                    );
                    enc.write_timestamp(qs, 12);
                    dispatch(&mut enc, &pl3, &bg3b, 1);
                    enc.write_timestamp(qs, 13);

                    // Readback copies
                    enc.copy_buffer_to_buffer(
                        &forces_buf,
                        0,
                        &rb_forces,
                        0,
                        layout.output_stride_bytes * n as u64,
                    );
                    enc.copy_buffer_to_buffer(&epa_buf, 0, &rb_epa, 0, 4 * n as u64);
                    enc.copy_buffer_to_buffer(&energy_buf, 0, &rb_etot, 0, 4);

                    // Resolve timestamps 4..14
                    enc.resolve_query_set(qs, 4..N_TS, rb, 0);
                    enc.copy_buffer_to_buffer(
                        rb,
                        0,
                        ts_rb.as_ref().unwrap(),
                        0,
                        8 * (N_TS - 4) as u64,
                    );
                } else {
                    dispatch(&mut enc, &pl0c, &bg0c, n);
                    dispatch(&mut enc, &pl1, &bg1, n);
                    dispatch(&mut enc, &pl2, &bg2, n);
                    dispatch(&mut enc, &pl3, &bg3a, n);
                    dispatch(&mut enc, &pl3, &bg3b, 1);
                    enc.copy_buffer_to_buffer(
                        &forces_buf,
                        0,
                        &rb_forces,
                        0,
                        layout.output_stride_bytes * n as u64,
                    );
                    enc.copy_buffer_to_buffer(&epa_buf, 0, &rb_epa, 0, 4 * n as u64);
                    enc.copy_buffer_to_buffer(&energy_buf, 0, &rb_etot, 0, 4);
                }

                let cpu_rb = Instant::now();
                self.queue.submit([enc.finish()]);
                self.device.poll(wgpu::Maintain::Wait);
                t.readback_cpu_ns = cpu_rb.elapsed().as_nanos() as f64;
            }

            // Read submission-B timestamps
            if self.has_ts {
                let raw_view = map_read_sync(&self.device, ts_rb.as_ref().unwrap());
                let raw: &[u64] = bytemuck::cast_slice(&raw_view);
                let p = self.ts_period;
                // raw[0] = slot4, raw[1] = slot5, ... raw[9] = slot13
                t.pass0c_ns = ts_delta_ns(raw, 0, 1, p);
                t.pass1_ns = ts_delta_ns(raw, 2, 3, p);
                t.pass2_ns = ts_delta_ns(raw, 4, 5, p);
                t.pass3a_ns = ts_delta_ns(raw, 6, 7, p);
                t.pass3b_ns = ts_delta_ns(raw, 8, 9, p);
                drop(raw_view);
                ts_rb.as_ref().unwrap().unmap();
            }

            // readback buffers (forces/epa/etot) are only copied, not inspected here.
            // They are CPU_READ-mapped implicitly by poll(Wait); no explicit unmap needed.

            results.push(t);
        }
        results
    }

    fn print_cellist_report(&self, rep: usize) {
        let (_, _, box_l) = fcc_supercell(rep);
        let n = rep * rep * rep * 4;
        let nc = n_cells_from_box(box_l, CELL_SIZE);

        print!("  Warming up  N={n:<5}  ({WARMUP} iterations)... ");
        let _ = self.profile_cellist(rep, WARMUP);
        println!("done");

        print!("  Measuring   N={n:<5}  ({MEASURE} iterations)... ");
        let samples = self.profile_cellist(rep, MEASURE);
        println!("done");

        // Compute medians per field
        macro_rules! median {
            ($field:ident) => {{
                let mut v: Vec<f64> = samples.iter().map(|s| s.$field).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            }};
        }
        let p0a = median!(pass0a_ns);
        let p0b = median!(pass0b_ns);
        let cpu = median!(cpu_prefix_ns);
        let p0c = median!(pass0c_ns);
        let p1 = median!(pass1_ns);
        let p2 = median!(pass2_ns);
        let p3a = median!(pass3a_ns);
        let p3b = median!(pass3b_ns);
        let rb = median!(readback_cpu_ns);

        let gpu_cl = p0a + p0b + p0c;
        let gpu_phys = p1 + p2 + p3a + p3b;
        let total = gpu_cl + cpu + gpu_phys + rb;

        println!();
        println!("  ┌─ N={n}  grid={nc}³  box={box_l:.1}Å ──────────────────────────────────────────────┐");
        println!("  │  Cell list construction (GPU)                                                    │");
        println!(
            "  │    pass0a  Morton ID assign       {:>12}  ({:4.1}%)                        │",
            fmt_ns(p0a),
            100. * p0a / total
        );
        println!(
            "  │    pass0b  atomicAdd counts        {:>12}  ({:4.1}%)                        │",
            fmt_ns(p0b),
            100. * p0b / total
        );
        println!(
            "  │    ── CPU  prefix-sum + upload ──  {:>12}  ({:4.1}%)  ← BOTTLENECK           │",
            fmt_ns(cpu),
            100. * cpu / total
        );
        println!(
            "  │    pass0c  scatter sorted_atoms    {:>12}  ({:4.1}%)                        │",
            fmt_ns(p0c),
            100. * p0c / total
        );
        println!("  │  EAM physics (GPU)                                                               │");
        println!(
            "  │    pass1   density accumulation    {:>12}  ({:4.1}%)                        │",
            fmt_ns(p1),
            100. * p1 / total
        );
        println!(
            "  │    pass2   forces + energy         {:>12}  ({:4.1}%)                        │",
            fmt_ns(p2),
            100. * p2 / total
        );
        println!(
            "  │    pass3a  reduce atoms→partials   {:>12}  ({:4.1}%)                        │",
            fmt_ns(p3a),
            100. * p3a / total
        );
        println!(
            "  │    pass3b  reduce partials→total   {:>12}  ({:4.1}%)                        │",
            fmt_ns(p3b),
            100. * p3b / total
        );
        println!(
            "  │  Readback (CPU map+poll)           {:>12}  ({:4.1}%)                        │",
            fmt_ns(rb),
            100. * rb / total
        );
        println!(
            "  │  ─────────────────────────────────────────────────────────────────────────────  │"
        );
        println!(
            "  │  GPU cell-list total               {:>12}  ({:4.1}%)                        │",
            fmt_ns(gpu_cl),
            100. * gpu_cl / total
        );
        println!(
            "  │  GPU physics total                 {:>12}  ({:4.1}%)                        │",
            fmt_ns(gpu_phys),
            100. * gpu_phys / total
        );
        println!(
            "  │  Frame total (median)              {:>12}                                    │",
            fmt_ns(total)
        );
        println!("  └──────────────────────────────────────────────────────────────────────────────────┘");
        println!();

        if !self.has_ts {
            println!("  ⚠  GPU timestamps unavailable — timings above are wall-clock (Instant).");
            println!("     On llvmpipe this is still accurate (synchronous execution).");
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────
fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let profiler = pollster::block_on(Profiler::new());

    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║         CREAM GPU Per-Pass Profiling  (CellList, synthetic Cu EAM)             ║");
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
    println!("Measurement: {WARMUP} warmup + {MEASURE} measured iterations; median reported.");
    println!(
        "Timestamps:  GPU QuerySet (ns/tick = {:.2})",
        profiler.ts_period
    );
    println!();

    for &rep in REPS {
        profiler.print_cellist_report(rep);
    }

    println!("Legend:");
    println!("  pass0a   Morton ID assignment (1 GPU dispatch)");
    println!("  pass0b   atomicAdd per-cell count (1 GPU dispatch)");
    println!("  CPU      map readback + prefix sum + re-upload  ← only CPU-GPU round-trip");
    println!("  pass0c   scatter atoms into sorted_atoms[] (1 GPU dispatch)");
    println!("  pass1    EAM density accumulation (27-cell neighbourhood)");
    println!("  pass2    EAM forces + per-atom energy (27-cell neighbourhood)");
    println!("  pass3a   GPU tree-reduction: N atoms → ceil(N/64) partial sums");
    println!("  pass3b   GPU tree-reduction: partials → 1 scalar total energy");
    println!("  Readback copy_buffer + map_async + unmap (3 buffers)");
}
