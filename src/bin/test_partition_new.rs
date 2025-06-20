use env_logger::Env;
use rustc_hash::FxHashSet;
use std::{time::Instant, path::Path};
use tucanos::{
    mesh::ConnectedComponents, Idx, Result, Tag,
};
use tucanos::mesh::test_meshes::test_mesh_3d;
use tucanos::metric::{AnisoMetric3d,Metric};
use tucanos::mesh::Point;
pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    init_log("error");
    
    // Dossier pour les sorties vtu 
    let output_dir = Path::new("load_balancing_hilbert");
     if !output_dir.exists() {
            std::fs::create_dir(output_dir)?;
        }
    // Load the mesh
    //let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb("data/simple3d.meshb")?;
    let mut mesh = test_mesh_3d().split().split().split();
    let n = mesh.n_elems() as usize;
    // Creating another Aniso_Metric 
    let v0 = Point::<3>::new(1.0, 0., 0.);
    let v1 = Point::<3>::new(0., 0.1, 0.);
    let v2 = Point::<3>::new(0., 0., 0.01);
    let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);
    assert!(f64::abs(m.vol() - 0.001) < 1e-12);
  
    // Partition Number
    let n_parts = 8;
    mesh.clear_all();
    println!("{}", mesh.n_elems());
    let now = Instant::now();
    mesh.reorder_hilbert();
    // Necessary to compute implied_metric.
    mesh.compute_vertex_to_elems();
    mesh.compute_volumes();
    // Making sure both metrics have the same size 
    let aniso_vec = vec![m;n];
    mesh.work_evaluation_aniso(&aniso_vec);
    mesh.partition_hilbert_ball(n_parts);
    mesh.partition_correction(n_parts);
    let t = now.elapsed().as_secs_f64();
    mesh.compute_face_to_elems();
    let q = mesh.partition_quality()?;

    let file_name = format!("hilbert_balanced.vtu"); 
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
    Ok(())
}
