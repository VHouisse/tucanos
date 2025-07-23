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
    mesh::{Point, SimplexMesh, Tetrahedron, test_meshes::test_mesh_3d},
    metric::IsoMetric,
    remesher::{ElementCostEstimator, TotoCostEstimator},
};
//use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

// pub fn load_costs_and_calculate_average(file_path: &Path) -> Result<Option<f64>> {
//     if !file_path.exists() {
//         return Ok(None); // Fichier non trouvé
//     }

//     let content = fs::read_to_string(file_path)?;
//     let costs: Vec<f64> = content
//         .lines()
//         .filter_map(|line| line.parse::<f64>().ok()) // Tente de parser chaque ligne en f64
//         .collect();

//     if costs.is_empty() {
//         Ok(None) // Le fichier est vide ou ne contient pas de nombres valides
//     } else {
//         let sum_costs: f64 = costs.iter().sum();
//         let average_cost = sum_costs / costs.len() as f64;
//         Ok(Some(average_cost))
//     }
// }
const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const CENTER_Z: f64 = 0.3;
const RADIUS: f64 = 0.1;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * RADIUS;
const H_INSIDE_SPHERE_ISO: f64 = 0.005;
const H_OUTSIDE_SPHERE_ISO: f64 = 0.1;

fn calculate_iso_metric(p: Point<3>) -> f64 {
    let mut res = H_OUTSIDE_SPHERE_ISO;
    let x = p[0];
    let y = p[1];
    let z = p[2];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_SPHERE_ISO;
    }
    res
}
// const H_INSIDE_SPHERE_ANISO: f64 = 0.01;
// const H_OUTSIDE_SPHERE_ANISO: f64 = 0.2;
// /// Calculates the anisotropic metric value for a given point.
// fn calculate_aniso_metric(p: Point<3>) -> f64 {
//     let mut res = H_OUTSIDE_SPHERE_ANISO;
//     let x = p[0];
//     let y = p[1];
//     let z = p[2];
//     let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);

//     if dist_sq <= RADIUS_SQ_ACTUAL {
//         res = H_INSIDE_SPHERE_ANISO;
//     }
//     res * (1.0 + 0.5 * (x - CENTER_X).abs())
// }

fn print_partition_cc(msh: &SimplexMesh<3, Tetrahedron>, n_parts: usize) {
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
    let output_dir = Path::new("Global_Partitionnement");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }

    let mut msh = test_mesh_3d().split().split().split();

    println!("# of elements: {}", msh.n_elems());

    let n_parts = 4;
    let m: Vec<IsoMetric<3>> = msh
        .verts()
        .map(|v| IsoMetric::<3>::from(calculate_iso_metric(v)))
        .collect();
    msh.compute_volumes();
    let estimator = TotoCostEstimator::<3, Tetrahedron, IsoMetric<3>>::new(&m);
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

    // let start = Instant::now();
    // let (quality, imbalance) = msh.partition::<RCMPartitioner>(n_parts, None)?;
    // let t = start.elapsed();
    // println!(
    //     "RCMPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
    //     t.as_secs_f64(),
    //     quality,
    //     imbalance
    // );
    // print_partition_cc(&msh, n_parts);

    // let start = Instant::now();
    // let (quality, imbalance) = msh.partition::<KMeansPartitioner3d>(n_parts, None)?;
    // let t = start.elapsed();
    // println!(
    //     "KMeansPartitioner3d: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
    //     t.as_secs_f64(),
    //     quality,
    //     imbalance
    // );
    // print_partition_cc(&msh, n_parts);
    Ok(())
}
