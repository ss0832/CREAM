//! Helper example: list all GPU adapters wgpu can see.
//! Use this to confirm your GPU is detected before running tests.
//!
//! ```bash
//! cargo run --example list_adapters
//! ```

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            println!("No GPU adapters found.");
            println!("On Linux, install Vulkan drivers (e.g. mesa-vulkan-drivers).");
            println!("On Windows/Mac, ensure your GPU drivers are up to date.");
            return;
        }
        println!("Found {} adapter(s):", adapters.len());
        for (i, a) in adapters.iter().enumerate() {
            let info = a.get_info();
            println!(
                "  [{i}] {:?}  backend={:?}  device_type={:?}",
                info.name, info.backend, info.device_type
            );
        }
    });
}
