//! Mesh partition example
use env_logger::Env;
use std::{path::Path, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Result,
    mesh::{
        Mesh,
        partition::{
            BFSPartitionner, BFSWRPartitionner, HilbertBallPartitioner, HilbertPartitioner,
        },
    },
};
use tucanos::{
    mesh::{Point, SimplexMesh, Triangle, test_meshes::test_mesh_2d},
    metric::IsoMetric,
    remesher::{ElementCostEstimator, NoCostEstimator},
};
//use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}
const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const RADIUS: f64 = 0.1;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * RADIUS;
const H_INSIDE_CIRCLE_ISO: f64 = 0.005;
const H_OUTSIDE_CIRCLE_ISO: f64 = 0.1;

fn calculate_iso_metric(p: Point<2>) -> f64 {
    let mut res = H_OUTSIDE_CIRCLE_ISO;
    let x = p[0];
    let y = p[1];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_CIRCLE_ISO;
    }
    res
}

fn print_partition_cc(msh: &SimplexMesh<2, Triangle>, n_parts: usize) {
    for i in 0..n_parts {
        let pmesh = msh.get_partition(i).mesh;
        let faces = pmesh.all_faces();
        let graph = pmesh.element_pairs(&faces);
        let cc = graph.connected_components().unwrap();
        let n_cc = cc.iter().copied().max().unwrap_or(0) + 1;
        println!("  part {i}: {n_cc} components");
    }
}

pub fn main() -> Result<()> {
    let output_dir = Path::new("Global_Partitionnement_2D");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
    let mut msh = test_mesh_2d().split().split().split().split().split();

    let file_name = "Maillage_Iso.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    println!("# of elements: {}", msh.n_elems());

    let n_parts = 4;
    let m: Vec<IsoMetric<2>> = msh
        .verts()
        .map(|v| IsoMetric::<2>::from(calculate_iso_metric(v)))
        .collect();
    msh.compute_volumes();
    let estimator = NoCostEstimator::<2, Triangle, IsoMetric<2>>::new(&m);
    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertPartitioner>(n_parts, Some(weights))?;
    let file_name = "Partitionned_Hilbert.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    let t = start.elapsed();
    println!(
        "HilbertPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertBallPartitioner>(n_parts, Some(weights))?;
    let file_name = "Partitionned_Hilbert_Ball.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    let t = start.elapsed();
    println!(
        "HilbertBallPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<BFSPartitionner>(n_parts, Some(weights))?;
    let file_name = "Partitionned_BFS.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    let t = start.elapsed();
    println!(
        "BFSPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<BFSWRPartitionner>(n_parts, Some(weights))?;
    let file_name = "Partitionned_BFSWR.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    let t = start.elapsed();
    println!(
        "BFSWRPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);
    Ok(())
}
