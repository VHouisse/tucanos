use clap::Parser;
use std::{path::Path, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{Result, mesh::Mesh};
use tucanos::{
    geometry::LinearGeometry,
    mesh::{Edge, Point, Triangle, test_meshes::test_mesh_2d},
    metric::{AnisoMetric, AnisoMetric2d, IsoMetric},
    remesher::{NoCostEstimator, ParallelRemesher, ParallelRemesherParams, RemesherParams},
};
#[derive(Parser, Debug)]
#[command(author, version, about = "Effectue un remaillage 2D avec des options configurables.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 7)] // Default splits for 2D might be different
    num_splits: usize,

    #[arg(short, long, default_value_t = String::from("iso"))]
    metric_type: String,

    #[arg(short, long, default_value_t = 4)]
    n_parts: usize,

    #[arg(short, long, default_value_t = String::from("Toto"))]
    cost_estimator: String,
}

const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const RADIUS: f64 = 0.1; // Consistent naming with 3D
const RADIUS_SQ_ACTUAL: f64 = RADIUS * RADIUS; // Consistent naming with 3D

const H_INSIDE_CIRCLE_ISO: f64 = 0.001; // Consistent naming with 3D
const H_OUTSIDE_CIRCLE_ISO: f64 = 0.1; // Consistent naming with 3D
const H_INSIDE_CIRCLE_ANISO: f64 = 0.002; // Added for aniso example, adjust as needed
const H_OUTSIDE_CIRCLE_ANISO: f64 = 0.2; // Added for aniso example, adjust as needed

fn calculate_iso_metric_2d(p: Point<2>) -> f64 {
    let mut res = H_OUTSIDE_CIRCLE_ISO;
    let x = p[0];
    let y = p[1];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_CIRCLE_ISO;
    }
    res
}

/// Calculates the anisotropic metric value for a given 2D point.
fn calculate_aniso_metric_2d(p: Point<2>) -> f64 {
    let mut res = H_OUTSIDE_CIRCLE_ANISO;
    let x = p[0];
    let y = p[1];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_CIRCLE_ANISO;
    }
    // Simple anisotropic factor for demonstration, adjust as needed
    res * (1.0 + 0.5 * (x - CENTER_X).abs())
}
fn main() -> Result<()> {
    let args = Args::parse();

    let output_dir = Path::new("Results_For_Remeshing_2d");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
    let mut msh = test_mesh_2d();
    for _ in 0..args.num_splits {
        msh = msh.split();
    }
    println!(
        "Nombre d'éléments contenus dans le maillage {} ",
        msh.n_elems()
    );
    match args.metric_type.as_str() {
        "iso" => {
            let m: Vec<IsoMetric<2>> = msh
                .verts()
                .map(|v| IsoMetric::<2>::from(calculate_iso_metric_2d(v)))
                .collect();
            let (bdy, _) = msh.boundary();
            let _topo = msh.compute_topology();
            let geom = LinearGeometry::<2, Edge>::new(&msh, bdy).unwrap();
            msh.compute_volumes();
            let remesher = ParallelRemesher::<
                2,
                Triangle,
                IsoMetric<2>,
                tmesh::mesh::partition::HilbertPartitioner,
                NoCostEstimator<2, Triangle, IsoMetric<2>>,
            >::new(msh, m, 8)?;
            let file_name = "Partitionned_Hilbert.vtu".to_string();
            let output_path = output_dir.join(&file_name);
            remesher
                .partitionned_mesh()
                .write_vtk(output_path.to_str().unwrap())?;
            let dd_params = ParallelRemesherParams::default();
            let params = RemesherParams::default();
            let time = Instant::now();
            (msh, _, _) = remesher.remesh(&geom, params, &dd_params).unwrap();
            let t2 = time.elapsed();
            println!("Temps de remaillage Avec Estimation du travail {t2:?}");
            let file_name = "Remeshed_Hilbert.vtu".to_string();
            let output_path = output_dir.join(&file_name);
            msh.write_vtk(output_path.to_str().unwrap())?;
        }
        "aniso" => {
            let m: Vec<AnisoMetric2d> = msh
                .verts()
                .map(|v| {
                    AnisoMetric2d::from_iso(&IsoMetric::<2>::from(calculate_aniso_metric_2d(v)))
                })
                .collect();
            let (bdy, _) = msh.boundary();
            let _topo = msh.compute_topology();
            let geom = LinearGeometry::<2, Edge>::new(&msh, bdy).unwrap();
            msh.compute_volumes();
            let remesher = ParallelRemesher::<
                2,
                Triangle,
                AnisoMetric2d,
                tmesh::mesh::partition::HilbertPartitioner,
                NoCostEstimator<2, Triangle, AnisoMetric2d>,
            >::new(msh, m, 8)?;
            let file_name = "Partitionned_Hilbert.vtu".to_string();
            let output_path = output_dir.join(&file_name);
            remesher
                .partitionned_mesh()
                .write_vtk(output_path.to_str().unwrap())?;
            let dd_params = ParallelRemesherParams::default();
            let params = RemesherParams::default();
            let time = Instant::now();
            (msh, _, _) = remesher.remesh(&geom, params, &dd_params).unwrap();
            let t2 = time.elapsed();
            println!("Temps de remaillage Avec Estimation du travail {t2:?}");
            let file_name = "Remeshed_Hilbert.vtu".to_string();
            let output_path = output_dir.join(&file_name);
            msh.write_vtk(output_path.to_str().unwrap())?;
        }
        _ => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type de métrique non valide",
            )));
        }
    }

    Ok(())
}
