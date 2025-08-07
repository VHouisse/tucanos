//! Mesh partition example
use clap::Parser;

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
    mesh::{
        Elem, GElem, HasTmeshImpl, Point, SimplexMesh, Tetrahedron, Triangle,
        test_meshes::test_mesh_3d,
    },
    metric::{AnisoMetric, AnisoMetric3d, HasImpliedMetric, IsoMetric, Metric},
    remesher::{
        ElementCostEstimator, NoCostEstimator, ParallelRemesher, ParallelRemesherParams,
        RemesherParams, TotoCostEstimator,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Effectue un remaillage 3D avec des options configurables.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    num_splits: usize,

    #[arg(short, long, default_value_t = String::from("iso"))]
    metric_type: String,

    #[arg(short, long, default_value_t = 8)]
    n_parts: u32,

    #[arg(short, long, default_value_t = String::from("Toto"))]
    cost_estimator: String,

    #[arg(short, long, default_value_t = String::from("HilbertBallPartitionner"))]
    partitionner: String,
}

const CENTER_X: f64 = 0.4;
const CENTER_Y: f64 = 0.4;
const CENTER_Z: f64 = 0.4;
const RADIUS: f64 = 1.0;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * 0.2;
const H_INSIDE_SPHERE_ISO: f64 = 0.01;
const H_OUTSIDE_SPHERE_ISO: f64 = 3.0;
const H_INSIDE_SPHERE_ANISO: f64 = 0.01;
const H_OUTSIDE_SPHERE_ANISO: f64 = 0.2;

// fn calculate_split_metric(mesh: &SimplexMesh<3, Tetrahedron>) -> Vec<AnisoMetric3d> {
//     let mut result_metrics = vec![AnisoMetric3d::default(); mesh.n_verts() as usize];
//     let verts: Vec<usize> = (0..mesh.n_verts() as usize).collect();
//     let e2e = mesh.get_vertex_to_elems().unwrap();
//     for i_vert in verts {
//         let elems = e2e.row(i_vert);
//         let _gelem = mesh.gelem(mesh.elem(elems[0] as u32));
//         let mut chosen_metric = AnisoMetric3d::from_iso(&IsoMetric::<3>::from(H_INSIDE_SPHERE_ISO));
//         let p = mesh.vert(i_vert as u32);
//         let x = p[0];
//         let y = p[1];
//         let z = p[2];
//         let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);
//         if dist_sq <= RADIUS_SQ_ACTUAL {
//             chosen_metric = AnisoMetric3d::from_iso(&IsoMetric::<3>::from(H_INSIDE_SPHERE_ISO));
//         }

//         // }AnisoMetric3d::from_iso(&IsoMetric::<3>::from(H_INSIDE_SPHERE_ISO));
//         result_metrics[i_vert] = chosen_metric;
//     }
//     result_metrics
// }
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
// #[allow(clippy::unnecessary_wraps)]
// #[allow(clippy::extra_unused_type_parameters)]
// #[allow(clippy::needless_pass_by_value)]
// fn perform_remeshing_sequential<
//     const D: usize,
//     E: Elem,
//     M: Metric<D>
//         + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
//     P: Partitioner,
//     C: ElementCostEstimator<D, E, M>,
// >(
//     mut msh: SimplexMesh<D, E>,
//     geom: &LinearGeometry<D, E::Face>,
//     metrics: Vec<M>,
//     cost_estimator_name: &str, // Renamed to avoid confusion with the generic type C
//     partitioner_name: &str,    // Added partitioner name for printout
//     _n_parts: u32,
//     metric_type_arg: &str,
// ) -> Result<SimplexMesh<D, E>>
// where
//     SimplexMesh<D, E>: HasTmeshImpl<D, E>,
//     SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face> + HasCurvature<D>,
//     E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
// {
//     msh.compute_volumes();
//     let n_elements = msh.n_elems();
//     let n_verts_init = msh.n_verts();
//     // let remesher = ParallelRemesher::<D, E, M, P, C>::new(msh, metrics, n_parts)?;
//     // let dd_params = ParallelRemesherParams::default();
//     let mut remesher = Remesher::new(&msh, &metrics, geom).unwrap();
//     let params = RemesherParams::default();
//     let now = Instant::now();
//     let stats = remesher.remesh(&params, geom).unwrap();
//     let infos = ParallelRemeshingInfo {
//         info: RemeshingInfo {
//             n_verts_init,
//             n_verts_final: 100,
//             time: now.elapsed().as_secs_f64(),
//             remesh_stats: stats,
//         },
//         ..Default::default()
//     };
//     infos.info.print_summary_remesh_stats();
//     let t2 = now.elapsed();
//     debug!(
//         "DATA,D={D},metric_type={metric_type_arg},cost_estimator={cost_estimator_name},partitioner={partitioner_name},num_elements={n_elements},time_seconds={t2:?}"
//     );
//     let new_mesh = remesher.to_mesh(false);
//     let n_m_elements = new_mesh.n_elems();
//     let n_m_verts = new_mesh.n_verts();
//     println!(
//         "Nombre d'éléments avant : {n_elements}, nb_verts : {n_verts_init}, nb elems after {n_m_elements}, nb verts after {n_m_verts}",
//     );
//     Ok(new_mesh)
// }

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
    cost_estimator_name: &str, // Renamed to avoid confusion with the generic type C
    partitioner_name: &str,    // Added partitioner name for printout
    n_parts: u32,
    metric_type_arg: &str,
) -> Result<SimplexMesh<D, E>>
where
    SimplexMesh<D, E>: HasTmeshImpl<D, E>,
    SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face> + HasCurvature<D>,
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    msh.compute_volumes();
    let n_elements = msh.n_elems();
    let remesher = ParallelRemesher::<D, E, M, P, C>::new(msh, metrics, n_parts)?;
    let dd_params = ParallelRemesherParams::default();
    let params = RemesherParams::default();
    let time = Instant::now();
    let (meshed, info, _) = remesher.remesh(geom, params, &dd_params).unwrap();
    let total_elapsed_time = time.elapsed();
    let remeshing_time = info.remeshing_time;
    let remeshing_imbalance = info.remeshing_time_imbalance;
    println!(
        "DATA,D={D},metric_type={metric_type_arg},cost_estimator={cost_estimator_name},partitioner={partitioner_name},num_elements={n_elements},remeshing_time = {remeshing_time}, remeshing_imbalance = {remeshing_imbalance}, total_elapsed_time={total_elapsed_time:?}"
    );
    Ok(meshed)
}

#[allow(clippy::too_many_lines)]
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
    let mesh_elems = msh.n_elems();
    println!("Nombre d'éléments initiaux {mesh_elems}");
    let (bdy, _) = msh.boundary();
    let _topo = msh.compute_topology();
    let geom = LinearGeometry::<3, Triangle>::new(&msh, bdy).unwrap();
    msh.compute_volumes();
    let file_name = "Initial_Mesh.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    match args.metric_type.as_str() {
        "iso" => {
            let m: Vec<IsoMetric<3>> = msh
                .verts()
                .map(|v| IsoMetric::<3>::from(calculate_iso_metric(v)))
                .collect();

            if args.cost_estimator.as_str() == "Nocost" {
                match args.partitionner.as_str() {
                    "HilbertPartitionner" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            HilbertPartitioner,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "NoCost",
                            "HilbertPartitionner",
                            args.n_parts,
                            "iso",
                        )?;
                    }
                    "HilbertBallPartitionner" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            HilbertBallPartitioner,
                            NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            BFSPartitionner,
                            NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            BFSWRPartitionner,
                            NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisKWay>,
                            NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh, &geom, m, "Nocost", "MetisKWay", args.n_parts, "iso"
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisRecursive>,
                            NoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "MetisRecursive",
                            args.n_parts,
                            "iso",
                        )?;
                    }
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            HilbertPartitioner,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            HilbertBallPartitioner,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            BFSPartitionner,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            BFSWRPartitionner,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
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
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisKWay>,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh, &geom, m, "Toto", "MetisKWay", args.n_parts, "iso"
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisRecursive>,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh, &geom, m, "Toto", "MetisRecursive", args.n_parts, "iso"
                        )?;
                    }
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
            let _m: Vec<AnisoMetric3d> = msh
                .verts()
                .map(|v| AnisoMetric3d::from_iso(&IsoMetric::<3>::from(calculate_aniso_metric(v))))
                .collect();
            msh.compute_vertex_to_elems();
            let m = calculate_split_metric_elems(&msh);

            if args.cost_estimator.as_str() == "Nocost" {
                match args.partitionner.as_str() {
                    "HilbertBallPartitionner" => {
                        let msh: SimplexMesh<3, Tetrahedron> = perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            HilbertBallPartitioner,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "aniso",
                        )
                        .unwrap();
                        let file_name = "A_HB_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    "HilbertPartitionner" => {
                        let msh = perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            HilbertPartitioner,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "NoCost",
                            "HilbertPartitionner",
                            args.n_parts,
                            "aniso",
                        )
                        .unwrap();
                        let file_name = "A_H_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            BFSPartitionner,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
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
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            BFSWRPartitionner,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
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
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            MetisPartitioner<MetisKWay>,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh, &geom, m, "Nocost", "MetisKWay", args.n_parts, "aniso"
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            MetisPartitioner<MetisRecursive>,
                            NoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "MetisRecursive",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
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
                        let msh: SimplexMesh<3, Tetrahedron> = perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            HilbertPartitioner,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertPartitionner",
                            args.n_parts,
                            "aniso",
                        )
                        .unwrap();
                        let file_name = "A_H_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    "HilbertBallPartitionner" => {
                        let msh: SimplexMesh<3, Tetrahedron> = perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            HilbertBallPartitioner,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "HilbertBallPartitionner",
                            args.n_parts,
                            "aniso",
                        )
                        .unwrap();
                        let file_name = "A_H_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    "BFSPartitionner" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            BFSPartitionner,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
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
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            BFSWRPartitionner,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
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
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            MetisPartitioner<MetisKWay>,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh, &geom, m, "Toto", "MetisKWay", args.n_parts, "aniso"
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            MetisPartitioner<MetisRecursive>,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "MetisRecursive",
                            args.n_parts,
                            "aniso",
                        )?;
                    }
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
