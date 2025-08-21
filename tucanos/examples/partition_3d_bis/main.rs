//! Mesh partition example
use env_logger::Env;
use nalgebra::Vector3;
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
    mesh::{GElem, SimplexMesh, Tetrahedron, test_meshes::test_mesh_3d},
    metric::{AnisoMetric, AnisoMetric3d, IsoMetric},
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

// fn calculate_iso_metric(p: Point<3>) -> f64 {
//     let mut res = H_OUTSIDE_SPHERE_ISO;
//     let x = p[0];
//     let y = p[1];
//     let z = p[2];
//     let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);

//     if dist_sq <= RADIUS_SQ_ACTUAL {
//         res = H_INSIDE_SPHERE_ISO;
//     }
//     res
// }
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

const CENTER_X: f64 = 0.4;
const CENTER_Y: f64 = 0.4;
const CENTER_Z: f64 = 0.4;
const RADIUS: f64 = 1.0;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * 0.2;
const H_INSIDE_SPHERE_ISO: f64 = 0.01;

// const H_OUTSIDE_SPHERE_ISO: f64 = 3.0;

fn calculate_split_metric_elems(mesh: &SimplexMesh<3, Tetrahedron>) -> Vec<AnisoMetric3d> {
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);
    for g_elem in mesh.gelems() {
        let mut chosen_metric = g_elem.implied_metric();
        let p = g_elem.center();
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);
        if dist_sq <= RADIUS_SQ_ACTUAL {
            chosen_metric = AnisoMetric3d::from_iso(&IsoMetric::<3>::from(H_INSIDE_SPHERE_ISO));
        }
        result_metrics.push(chosen_metric);
    }

    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
}
const STRETCH_MAGNITUDE: f64 = 0.001;
const PERP_MAGNITUDE: f64 = 0.1;
fn calculate_stretching_metric(mesh: &SimplexMesh<3, Tetrahedron>) -> Vec<AnisoMetric3d> {
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);

    let stretch_direction = Vector3::new(0.0, 0.0, 1.0);
    let perp_direction_x = Vector3::new(1.0, 0.0, 0.0);
    let perp_direction_y = Vector3::new(0.0, 1.0, 0.0);

    for g_elem in mesh.gelems() {
        let mut chosen_metric = g_elem.implied_metric();

        let p = g_elem.center();
        let x = p[0];
        let y = p[1];
        // let z = p[2];
        if (x - 1.0).abs() < 0.1 && (y - 0.5).abs() < 0.3 {
            let dist_to_center_y = (p.y - 0.5).abs();
            let influence = 1.0 - (dist_to_center_y / 0.3);

            chosen_metric = AnisoMetric3d::from_sizes(
                &(stretch_direction * STRETCH_MAGNITUDE * influence),
                &(perp_direction_x * PERP_MAGNITUDE),
                &(perp_direction_y * PERP_MAGNITUDE),
            );
        }
        result_metrics.push(chosen_metric);
    }

    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
}

fn print_partition_cc(msh: &SimplexMesh<3, Tetrahedron>, n_parts: usize) {
    for i in 0..n_parts {
        let pmesh = msh.get_partition(i).mesh;
        let nb_elems = pmesh.n_elems();
        let n_verts = pmesh.n_verts();
        let faces = pmesh.all_faces();
        let graph = pmesh.element_pairs(&faces);
        let cc = graph.connected_components().unwrap();
        let n_cc = cc.iter().copied().max().unwrap_or(0) + 1;
        println!("  part {i}: {n_cc} components, nb_elems {nb_elems} n_verts {n_verts}");
    }
}

#[allow(clippy::too_many_lines)]
pub fn main() -> Result<()> {
    let output_dir = Path::new("Global_Partitionnement_3D");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }

    let mut msh = test_mesh_3d();
    for _ in 0..6 {
        msh = msh.split();
    }
    println!("# of elements: {}", msh.n_elems());
    let n_parts = 8;
    msh.compute_vertex_to_elems();
    msh.compute_volumes();
    let m = calculate_split_metric_elems(&msh);
    let _m = calculate_stretching_metric(&msh);
    msh.compute_volumes();

    let estimator = TotoCostEstimator::<3, Tetrahedron, AnisoMetric3d>::new(&m);
    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertPartitioner>(n_parts, Some(weights))?;
    let t = start.elapsed();
    let file_name = "Partitionned_Hilbert.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
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
    let t = start.elapsed();
    let file_name = "Partitionned_Hilbert_Ball.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
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
    let t = start.elapsed();
    let file_name = "Partitionned_BFS.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
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
    let t = start.elapsed();
    let file_name = "Partitionned_BFSWR.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
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
