use env_logger::Env;
use rustc_hash::FxHashSet;
use std::time::Instant;
use tucanos::{
    mesh::{PartitionType, ConnectedComponents}, Idx, Result, Tag,
};
use tucanos::mesh::test_meshes::test_mesh_3d;
use std::path::Path; 
pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    init_log("error");
    let mut renumbering :bool = true;
    // Dossier pour les sorties vtu 
    let mut output_dir = Path::new("mesh_partitions_renumbered");
    for i in 0..2 {
       
        if i == 1{renumbering=false;
                  output_dir = Path::new("mesh_partitions_classic");
                }
        if !output_dir.exists() {
            std::fs::create_dir(output_dir)?;
        }
        let mut mesh = test_mesh_3d().split().split().split();

        let n_parts = 8;

        let ptypes = [
            PartitionType::Hilbert(n_parts),
            //PartitionType::Scotch(n_parts),
            PartitionType::MetisRecursive(n_parts),
            PartitionType::MetisKWay(n_parts),
            
        ];

        for ptype in ptypes {
            
            println!("{ptype:?}");
            mesh.clear_all();
            let now = Instant::now();
            if renumbering{
                mesh.reorder_hilbert();
            }
            mesh.partition(ptype)?;
            let t = now.elapsed().as_secs_f64();
            mesh.compute_face_to_elems();
            let q = mesh.partition_quality()?;
            let file_name = format!("{}.vtu", ptype.to_string()); 
            let output_path = output_dir.join(&file_name);
            let _ = mesh.write_vtk(output_path.to_str().unwrap(),None, None);
            
            println!("elapsed time: {t:.2e}s");
            println!("quality: {q:.2e}");
            for i_part in 0..n_parts {
                let mut smsh = mesh.extract_tag(i_part as Tag + 1);
                let e2e = smsh.mesh.compute_elem_to_elems();
                let cc = ConnectedComponents::<Idx>::new(e2e);
                if cc.is_ok() {
                    let n_cc = cc?.tags().iter().clone().collect::<FxHashSet<_>>().len();
                    println!(
                        "partition {i_part} : {} elements,  {n_cc} connected components",
                        smsh.mesh.n_elems()
                    );
                } else {
                    println!(
                        "partition {i_part} : {} elements,  too many connected components",
                        smsh.mesh.n_elems()
                    );
                }
            }
        }
    }
    Ok(())
}
