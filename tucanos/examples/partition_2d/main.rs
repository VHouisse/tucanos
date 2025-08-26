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
    mesh::{GElem, SimplexMesh, Triangle, test_meshes::test_mesh_2d},
    metric::{AnisoMetric, AnisoMetric2d, IsoMetric},
    remesher::{ElementCostEstimator, NoCostEstimator},
};
//use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}
const CENTER_X: f64 = 0.4;
const CENTER_Y: f64 = 0.4;
const RADIUS: f64 = 1.0;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * 0.2;
const H_INSIDE_SPHERE_ISO: f64 = 0.01;

fn calculate_split_metric_elems(mesh: &SimplexMesh<2, Triangle>) -> Vec<AnisoMetric2d> {
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);
    for g_elem in mesh.gelems() {
        let mut chosen_metric = g_elem.implied_metric();
        let p = g_elem.center();
        let x = p[0];
        let y = p[1];

        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);
        if dist_sq <= RADIUS_SQ_ACTUAL {
            chosen_metric = AnisoMetric2d::from_iso(&IsoMetric::<2>::from(H_INSIDE_SPHERE_ISO));
        }
        result_metrics.push(chosen_metric);
    }

    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
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
#[allow(clippy::too_many_lines)]
pub fn main() -> Result<()> {
    let output_dir = Path::new("Global_Partitionnement_2D");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
    let mut msh = test_mesh_2d().split().split().split().split().split();
    msh.compute_vertex_to_elems();
    msh.compute_volumes();

    let file_name = "Maillage_Iso.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    println!("# of elements: {}", msh.n_elems());

    let n_parts = 4;
    let m: Vec<AnisoMetric2d> = calculate_split_metric_elems(&msh);
    let estimator = NoCostEstimator::<2, Triangle, AnisoMetric2d>::new(&m);
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

    #[cfg(feature = "metis")]
    {
        let weights = estimator.compute(&msh, &m);
        let start = Instant::now();
        let (quality, imbalance) =
            msh.partition::<MetisPartitioner<MetisRecursive>>(n_parts, Some(weights))?;
        let t = start.elapsed();
        let file_name = "Partitionned_MetisRecursive.vtu".to_string();
        let output_path = output_dir.join(&file_name);
        let _ = msh.write_vtk(output_path.to_str().unwrap());
        println!(
            "Metis Recursive: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
            t.as_secs_f64(),
            quality,
            imbalance
        );
        print_partition_cc(&msh, n_parts);
    }

    #[cfg(feature = "metis")]
    {
        let weights = estimator.compute(&msh, &m);
        let start = Instant::now();
        let (quality, imbalance) =
            msh.partition::<MetisPartitioner<MetisKWay>>(n_parts, Some(weights))?;
        let t = start.elapsed();
        let file_name = "Partitionned_MetisKWay.vtu".to_string();
        let output_path = output_dir.join(&file_name);
        let _ = msh.write_vtk(output_path.to_str().unwrap());
        println!(
            "MetisKWayPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
            t.as_secs_f64(),
            quality,
            imbalance
        );
        print_partition_cc(&msh, n_parts);
    }
    Ok(())
}
