//! Mesh partition example
use clap::Parser;
#[cfg(feature = "metis")]
use clap::Parser;
use log::debug;
use std::{path::Path, time::Instant};

#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Result,
    mesh::{
        Mesh,
        partition::{
            BFSPartitionner, BFSWRPartitionner, HilbertBallPartitioner, HilbertPartitioner,
            Partitioner,
        },
    },
};

use tucanos::{
    geometry::{LinearGeometry, curvature::HasCurvature},
    mesh::{Edge, Elem, HasTmeshImpl, Point, SimplexMesh, Triangle, test_meshes::test_mesh_2d},
    metric::{AnisoMetric, AnisoMetric2d, HasImpliedMetric, IsoMetric, Metric},
    remesher::{
        ElementCostEstimator, NoCostEstimator, ParallelRemeshingInfo, Remesher, RemesherParams,
        RemeshingInfo, TotoCostEstimator,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Effectue un remaillage 2D avec des options configurables.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    num_splits: usize,

    #[arg(short, long, default_value_t = String::from("iso"))]
    metric_type: String,

    #[arg(short, long, default_value_t = 32)]
    n_parts: u32,

    #[arg(short, long, default_value_t = String::from("Toto"))]
    cost_estimator: String,

    #[arg(short, long, default_value_t = String::from("HilbertBallPartitionner"))]
    partitionner: String,
}

const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const RADIUS: f64 = 0.2;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * 1.0;
const H_INSIDE_CIRCLE_ISO: f64 = 1.0;
const H_OUTSIDE_CIRCLE_ISO: f64 = 0.05;
const H_INSIDE_CIRCLE_ANISO: f64 = 0.01;
const H_OUTSIDE_CIRCLE_ANISO: f64 = 0.2;

fn calculate_split_metric(
    mesh: &SimplexMesh<2, Triangle>, // Spécifier D=2 et E=Triangle
) -> Vec<AnisoMetric2d> {
    let mut result_metrics = vec![AnisoMetric2d::default(); mesh.n_verts() as usize];
    let verts: Vec<usize> = (0..mesh.n_verts() as usize).collect();
    let e2e = mesh.get_vertex_to_elems().unwrap();
    for i_vert in verts {
        let elems = e2e.row(i_vert);
        let _gelem = mesh.gelem(mesh.elem(elems[0] as u32));
        let mut chosen_metric =
            AnisoMetric2d::from_iso(&IsoMetric::<2>::from(H_OUTSIDE_CIRCLE_ISO));
        let p = mesh.vert(i_vert as u32);
        let x = p[0];
        let y = p[1];
        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);
        if dist_sq <= RADIUS_SQ_ACTUAL {
            chosen_metric = AnisoMetric2d::from_iso(&IsoMetric::<2>::from(H_INSIDE_CIRCLE_ISO));
        }
        result_metrics[i_vert] = chosen_metric;
    }
    result_metrics
}
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
/// Calculates the anisotropic metric value for a given point.
fn calculate_aniso_metric(p: Point<2>) -> f64 {
    let mut res = H_OUTSIDE_CIRCLE_ANISO;
    let x = p[0];
    let y = p[1];
    let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);

    if dist_sq <= RADIUS_SQ_ACTUAL {
        res = H_INSIDE_CIRCLE_ANISO;
    }
    res * (1.0 + 0.5 * (x - CENTER_X).abs())
}

#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::extra_unused_type_parameters)]
#[allow(clippy::needless_pass_by_value)]
fn perform_remeshing<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
    P: Partitioner,
    C: ElementCostEstimator<D, E, M>,
>(
    mut msh: SimplexMesh<D, E>,
    geom: &LinearGeometry<D, E::Face>,
    metrics: Vec<M>,
    cost_estimator_name: &str,
    partitioner_name: &str,
    _n_parts: u32,
    metric_type_arg: &str,
) -> Result<()>
where
    SimplexMesh<D, E>: HasTmeshImpl<D, E>,
    SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face> + HasCurvature<D>,
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    msh.compute_volumes();
    let n_elements = msh.n_elems();
    let n_verts_init = msh.n_verts();

    // let remesher = ParallelRemesher::<D, E, M, P, C>::new(msh, metrics, n_parts)?;
    // let dd_params = ParallelRemesherParams::default();
    let mut remesher = Remesher::new(&msh, &metrics, geom).unwrap();
    let params = RemesherParams::default();
    let now = Instant::now();
    let stats = remesher.remesh(&params, geom).unwrap();

    let infos = ParallelRemeshingInfo {
        info: RemeshingInfo {
            n_verts_init,
            n_verts_final: 100,
            time: now.elapsed().as_secs_f64(),
            remesh_stats: stats,
        },
        ..Default::default()
    };
    infos.info.print_summary_remesh_stats();
    let t2 = now.elapsed();

    debug!(
        "DATA,D={D},metric_type={metric_type_arg},cost_estimator={cost_estimator_name},partitioner={partitioner_name},num_elements={n_elements},time_seconds={t2:?}"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
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
    let (bdy, _) = msh.boundary();
    let _topo = msh.compute_topology();
    let geom = LinearGeometry::<2, Edge>::new(&msh, bdy).unwrap();
    msh.compute_volumes();

    match args.metric_type.as_str() {
        "iso" => {
            let m: Vec<IsoMetric<2>> = msh
                .verts()
                .map(|v| IsoMetric::<2>::from(calculate_iso_metric(v)))
                .collect();

            if args.cost_estimator.as_str() == "Nocost" {
                match args.partitionner.as_str() {
                    "HilbertPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            HilbertPartitioner,
                            NoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "HilbertPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "HilbertBallPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            HilbertBallPartitioner,
                            NoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            BFSPartitionner,
                            NoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "BFSPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "BFSWRPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            BFSWRPartitionner,
                            NoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "BFSWRPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    // #[cfg(feature = "metis")]
                    // "MetisKWay" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         IsoMetric<2>,
                    //         MetisKWay,
                    //         NoCostEstimator<2, Triangle, IsoMetric<2>>,
                    //     >(
                    //         msh, &geom, m, "Nocost", "MetisKWay", args.n_parts
                    //     )?;
                    // }
                    // #[cfg(feature = "metis")]
                    // "MetisRecursive" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         IsoMetric<2>,
                    //         MetisRecursive,
                    //         NoCostEstimator<2, Triangle, IsoMetric<2>>,
                    //     >(
                    //         msh, &geom, m, "Nocost", "MetisRecursive", args.n_parts
                    //     )?;
                    // }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            } else {
                // Default to TotoCostEstimator if not "Nocost"
                match args.partitionner.as_str() {
                    "HilbertPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            HilbertPartitioner,
                            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "HilbertBallPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            HilbertBallPartitioner,
                            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            BFSPartitionner,
                            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "BFSPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "BFSWRPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            IsoMetric<2>,
                            BFSWRPartitionner,
                            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "BFSWRPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    // #[cfg(feature = "metis")]
                    // "MetisKWay" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         IsoMetric<2>,
                    //         MetisKWay,
                    //         TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                    //     >(msh, &geom, m, "Toto", "MetisKWay", args.n_parts)?;
                    // }
                    // #[cfg(feature = "metis")]
                    // "MetisRecursive" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         IsoMetric<2>,
                    //         MetisRecursive,
                    //         TotoCostEstimator<2, Triangle, IsoMetric<2>>,
                    //     >(
                    //         msh, &geom, m, "Toto", "MetisRecursive", args.n_parts
                    //     )?;
                    // }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            }
        }
        "aniso" => {
            let _m: Vec<AnisoMetric2d> = msh
                .verts()
                .map(|v| AnisoMetric2d::from_iso(&IsoMetric::<2>::from(calculate_aniso_metric(v))))
                .collect();
            msh.compute_vertex_to_elems();
            let m = calculate_split_metric(&msh);

            if args.cost_estimator.as_str() == "Nocost" {
                match args.partitionner.as_str() {
                    "HilbertPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            HilbertPartitioner,
                            NoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "HilbertPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "HilbertBallPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            HilbertBallPartitioner,
                            NoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            BFSPartitionner,
                            NoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "BFSPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "BFSWRPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            BFSWRPartitionner,
                            NoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "BFSWRPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    // #[cfg(feature = "metis")]
                    // "MetisKWay" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         AnisoMetric2d,
                    //         MetisKWay,
                    //         NoCostEstimator<2, Triangle, AnisoMetric2d>,
                    //     >(
                    //         msh, &geom, m, "Nocost", "MetisKWay", args.n_parts
                    //     )?;
                    // }
                    // #[cfg(feature = "metis")]
                    // "MetisRecursive" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         AnisoMetric2d,
                    //         MetisRecursive,
                    //         NoCostEstimator<2, Triangle, AnisoMetric2d>,
                    //     >(
                    //         msh, &geom, m, "Nocost", "MetisRecursive", args.n_parts
                    //     )?;
                    // }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            } else {
                // Default to TotoCostEstimator if not "Nocost"
                match args.partitionner.as_str() {
                    "HilbertPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            HilbertPartitioner,
                            TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "HilbertBallPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            HilbertBallPartitioner,
                            TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            BFSPartitionner,
                            TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "BFSPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    "BFSWRPartitionner" => {
                        perform_remeshing::<
                            2,
                            Triangle,
                            AnisoMetric2d,
                            BFSWRPartitionner,
                            TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "BFSWRPartitionner",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
                    // #[cfg(feature = "metis")]
                    // "MetisKWay" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         AnisoMetric2d,
                    //         MetisKWay,
                    //         TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                    //     >(msh, &geom, m, "Toto", "MetisKWay", args.n_parts)?;
                    // }
                    // #[cfg(feature = "metis")]
                    // "MetisRecursive" => {
                    //     perform_remeshing::<
                    //         2,
                    //         Triangle,
                    //         AnisoMetric2d,
                    //         MetisRecursive,
                    //         TotoCostEstimator<2, Triangle, AnisoMetric2d>,
                    //     >(
                    //         msh, &geom, m, "Toto", "MetisRecursive", args.n_parts
                    //     )?;
                    // }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            }
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
