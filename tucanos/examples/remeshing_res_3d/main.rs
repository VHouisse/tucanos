//! Mesh partition example
use clap::Parser;
use std::{path::Path, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Result,
    mesh::{Mesh, partition::HilbertBallPartitioner},
};

use tucanos::{
    geometry::{LinearGeometry, curvature::HasCurvature},
    mesh::{
        Elem, HasTmeshImpl, Point, SimplexMesh, Tetrahedron, Triangle, test_meshes::test_mesh_3d,
    },
    metric::{AnisoMetric, AnisoMetric3d, HasImpliedMetric, IsoMetric, Metric},
    remesher::{
        NoCostEstimator, ParallelRemesher, ParallelRemesherParams, RemesherParams,
        TotoCostEstimator,
    },
};
#[derive(Parser, Debug)]
#[command(author, version, about = "Effectue un remaillage 3D avec des options configurables.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    num_splits: usize,

    #[arg(short, long, default_value_t = String::from("iso"))]
    metric_type: String,

    #[arg(short, long, default_value_t = 4)]
    n_parts: u32,

    #[arg(short, long, default_value_t = String::from("Toto"))]
    cost_estimator: String,
}
const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const CENTER_Z: f64 = 0.3;
const RADIUS: f64 = 0.1;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * RADIUS;
const H_INSIDE_SPHERE_ISO: f64 = 0.005;
const H_OUTSIDE_SPHERE_ISO: f64 = 0.1;
const H_INSIDE_SPHERE_ANISO: f64 = 0.01;
const H_OUTSIDE_SPHERE_ANISO: f64 = 0.2;

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

/// Calculates the anisotropic metric value for a given point.
fn calculate_aniso_metric(p: Point<3>) -> f64 {
    let mut res = H_OUTSIDE_SPHERE_ANISO;
    let x = p[0];
    let y = p[1];
    let z = p[2];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_SPHERE_ANISO;
    }
    res * (1.0 + 0.5 * (x - CENTER_X).abs())
}

fn perform_remeshing<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
>(
    mut msh: SimplexMesh<D, E>,
    geom: &LinearGeometry<D, E>,
    metrics: Vec<M>,
    output_dir: &Path,
    file_prefix: &str,
    metric_name: &str,
) -> Result<()>
where
    SimplexMesh<D, E>: HasCurvature<D> + HasTmeshImpl<D, E>,
    SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face> + HasCurvature<D>,
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    msh.compute_volumes();

    let remesher = ParallelRemesher::<
        D,
        E,
        M,
        tmesh::mesh::partition::HilbertPartitioner,
        NoCostEstimator<D, E, M>,
    >::new(msh, metrics, 8)?; // Number of threads

    let partitionned_file_name = format!("Partitionned_Hilbert_{}.vtu", file_prefix);
    let partitionned_output_path = output_dir.join(&partitionned_file_name);
    remesher
        .partitionned_mesh()
        .write_vtk(partitionned_output_path.to_str().unwrap())?;

    let dd_params = ParallelRemesherParams::default();
    let params = RemesherParams::default();
    let time = Instant::now();
    (msh, _, _) = remesher.remesh(geom, params, &dd_params).unwrap();
    let t2 = time.elapsed();
    println!(
        "Temps de remaillage ({}) Avec Estimation du travail {:?}",
        metric_name, t2
    );

    let remeshed_file_name = format!("Remeshed_Hilbert_{}.vtu", file_prefix);
    let remeshed_output_path = output_dir.join(&remeshed_file_name);
    msh.write_vtk(remeshed_output_path.to_str().unwrap())?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let output_dir = Path::new("Results_For_Remeshing_3d");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
    let mut msh = test_mesh_3d();
    for _ in 0..args.num_splits {
        msh = msh.split();
    }
    match args.metric_type.as_str() {
        "iso" => {
            let m: Vec<IsoMetric<3>> = msh
                .verts()
                .map(|v| IsoMetric::<3>::from(calculate_iso_metric(v)))
                .collect();
            let (bdy, _) = msh.boundary();
            let _topo = msh.compute_topology();
            let geom = LinearGeometry::<3, Triangle>::new(&msh, bdy).unwrap();
            msh.compute_volumes();
            let remesher = ParallelRemesher::<
                3,
                Tetrahedron,
                IsoMetric<3>,
                HilbertBallPartitioner,
                TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
            >::new(msh, m, args.n_parts)?;
            match args.cost_estimator.as_str() {
                "Nocost" => {
                    let remesher = ParallelRemesher::<
                        3,
                        Tetrahedron,
                        IsoMetric<3>,
                        HilbertBallPartitioner,
                        NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                    >::new(msh, m, args.n_parts)?;
                }
                _ => {}
            }

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
            let m: Vec<AnisoMetric3d> = msh
                .verts()
                .map(|v| AnisoMetric3d::from_iso(&IsoMetric::<3>::from(calculate_aniso_metric(v))))
                .collect();

            let (bdy, _) = msh.boundary();
            let _topo = msh.compute_topology();
            let geom = LinearGeometry::<3, Triangle>::new(&msh, bdy).unwrap();
            msh.compute_volumes();

            let remesher = ParallelRemesher::<
                3,
                Tetrahedron,
                AnisoMetric3d,
                HilbertBallPartitioner,
                TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
            >::new(msh, m, args.n_parts)?;
            match args.cost_estimator.as_str() {
                "Nocost" => {
                    let remesher = ParallelRemesher::<
                        3,
                        Tetrahedron,
                        AnisoMetric3d,
                        HilbertBallPartitioner,
                        NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                    >::new(msh, m, args.n_parts)?;
                }
                _ => {}
            }

            let file_name = "Partitionned_Hilbert_Aniso.vtu".to_string();
            let output_path = output_dir.join(&file_name);
            remesher
                .partitionned_mesh()
                .write_vtk(output_path.to_str().unwrap())?;

            let dd_params = ParallelRemesherParams::default();
            let params = RemesherParams::default();
            let time = Instant::now();
            (msh, _, _) = remesher.remesh(&geom, params, &dd_params).unwrap();
            let t2 = time.elapsed();
            println!("Temps de remaillage (Anisotrope) Avec Estimation du travail {t2:?}");
            let file_name = "Remeshed_Hilbert_Aniso.vtu".to_string();
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
