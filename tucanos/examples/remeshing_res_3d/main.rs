//! Mesh partition example
use clap::Parser;

use std::{collections::HashMap, path::Path, time::Instant};

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
        Elem, GElem, HasTmeshImpl, SimplexMesh, Tetrahedron, Triangle, test_meshes::test_mesh_3d,
    },
    metric::{AnisoMetric, AnisoMetric3d, HasImpliedMetric, IsoMetric, Metric},
    remesher::{
        ElementCostEstimator, NoCostEstimator, ParallelRemesher, ParallelRemesherParams,
        ParallelRemeshingInfo, Remesher, RemesherParams, RemeshingInfo, TotoCostEstimator,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Effectue un remaillage 3D avec des options configurables.", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 3)]
    num_splits: usize,

    #[arg(short, long, default_value_t = String::from("iso"))]
    metric_type: String,

    #[arg(short, long, default_value_t = 2)]
    n_parts: u32,

    #[arg(short, long, default_value_t = String::from("Toto"))]
    cost_estimator: String,

    #[arg(short, long, default_value_t = String::from("HilbertBallPartitionner"))]
    partitionner: String,

    #[arg(short, long,  default_value_t = String::from("true"))]
    option: String,
}

const CENTER_X: f64 = 0.4;
const CENTER_Y: f64 = 0.4;
const CENTER_Z: f64 = 0.4;
const RADIUS: f64 = 1.0;
const RADIUS_SQ_ACTUAL: f64 = RADIUS * 0.3;

fn calculate_op_metric_elems(
    mesh: &SimplexMesh<3, Tetrahedron>,
    option: bool,
) -> Vec<AnisoMetric3d> {
    let h_inside_sphere_iso = if option { 5.0 } else { 0.01 };
    println!("h : {h_inside_sphere_iso}");
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);
    for g_elem in mesh.gelems() {
        let mut chosen_metric = g_elem.implied_metric();
        let p = g_elem.center();
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);
        if dist_sq <= RADIUS_SQ_ACTUAL {
            if option {
                chosen_metric.scale_aniso(0.25);
            } else {
                chosen_metric.scale_aniso(4.0); //AnisoMetric3d::from_iso(&IsoMetric::<3>::from(h_inside_sphere_iso));
            }
        }
        result_metrics.push(chosen_metric);
    }
    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
}

fn calculate_iso_metric(mesh: &SimplexMesh<3, Tetrahedron>, h: f64) -> Vec<IsoMetric<3>> {
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);

    let min_h = h / 2.0;
    let max_h = h / 2.0;

    let num_steps = 1;
    let step_size = (max_h - min_h) / (f64::from(num_steps));

    let target_sizes: Vec<f64> = (0..num_steps)
        .map(|i| min_h + f64::from(i) * step_size)
        .collect();

    let mut final_target_sizes = Vec::with_capacity(mesh.n_elems() as usize);
    for i in 0..mesh.n_elems() {
        final_target_sizes.push(target_sizes[(i % (target_sizes.len() as u32)) as usize]);
    }
    println!("-- Proportions des tailles d'éléments --");
    let total_elements = final_target_sizes.len();
    let mut counts: HashMap<u64, u32> = HashMap::new();
    let scale_factor = 1_000_000.0;

    for &size in &final_target_sizes {
        let scaled_size = (size * scale_factor) as u64;
        *counts.entry(scaled_size).or_insert(0) += 1;
    }

    let mut sorted_keys: Vec<&u64> = counts.keys().collect();
    sorted_keys.sort();
    println!("H maillage: {h}");
    for key in sorted_keys {
        let count = counts.get(key).unwrap();
        let original_size = (*key as f64) / scale_factor;
        let proportion = (f64::from(*count)) / (total_elements as f64) * 100.0;
        println!(
            "Taille: {original_size:.6} | Occurrences: {count} | Proportion: {proportion:.2}%",
        );
    }
    println!("---------------------------------------");

    for h in final_target_sizes {
        result_metrics.push(IsoMetric::from(h));
    }
    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
}

fn calculate_stretching_metric(mesh: &SimplexMesh<3, Tetrahedron>) -> Vec<AnisoMetric3d> {
    let mut result_metrics = Vec::with_capacity(mesh.n_elems() as usize);
    let mut am: [f64; 6] = [0.0; 6];
    // let stretch_direction = Vector3::new(0.0, 0.0, 1.0);
    // let perp_direction_x = Vector3::new(1.0, 0.0, 0.0);
    // let perp_direction_y = Vector3::new(0.0, 1.0, 0.0);

    for g_elem in mesh.gelems() {
        let mut chosen_metric = g_elem.implied_metric();
        let mut metric_am = g_elem.implied_metric();
        let p = g_elem.center();
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);
        if dist_sq <= RADIUS_SQ_ACTUAL {
            metric_am.scale(4.0);
            am[..5].copy_from_slice(&chosen_metric.m[..5]);
            am[0] = metric_am.m[0];
            chosen_metric = AnisoMetric3d::from_slice(&am);
        }

        result_metrics.push(chosen_metric);
    }

    mesh.elem_data_to_vertex_data_metric(&result_metrics)
        .unwrap()
}

fn quality_remeshed<const D: usize, E: Elem>(remeshed: &SimplexMesh<D, E>) -> f64 {
    let mut qualities = Vec::with_capacity(remeshed.n_elems() as usize);
    for g_elem in remeshed.gelems() {
        let quality = g_elem.quality();
        qualities.push(quality);
    }
    qualities.iter().sum::<f64>() / f64::from(remeshed.n_elems())
}

#[allow(clippy::too_many_arguments)]
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
    option: bool,
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
    let mean_quality = quality_remeshed(&meshed);
    println!("Mean quality of the remeshed mesh {mean_quality}");
    let total_elapsed_time = time.elapsed();
    let remeshing_partition_time = info.remeshing_partition_time;
    let remeshing_ifc_time = info.remeshing_ifc_time;
    let remeshing_imbalance = info.remeshing_ptime_imbalance;
    println!(
        "DATA,D={D},metric_type={metric_type_arg},cost_estimator={cost_estimator_name},partitioner={partitioner_name},num_elements={n_elements},
        remeshing_partition_time = {remeshing_partition_time:.2e},remeshing_ifc_time={remeshing_ifc_time:.2e}, remeshing_ptime_imbalance = {remeshing_imbalance},
        total_elapsed_time={total_elapsed_time:?},option={option}"
    );
    Ok(meshed)
}

#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::extra_unused_type_parameters)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_arguments)]
fn perform_remeshing_sequential<
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
    _cost_estimator_name: &str, // Renamed to avoid confusion with the generic type C
    _partitioner_name: &str,    // Added partitioner name for printout
    _n_parts: u32,
    _metric_type_arg: &str,
    _option: bool,
) -> Result<SimplexMesh<D, E>>
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
    // let t2 = now.elapsed();
    // debug!(
    //     "DATA,D={D},metric_type={metric_type_arg},cost_estimator={cost_estimator_name},partitioner={partitioner_name},num_elements={n_elements},time_seconds={t2:?}"
    // );
    let new_mesh = remesher.to_mesh(false);
    let n_m_elements = new_mesh.n_elems();
    let n_m_verts = new_mesh.n_verts();
    println!(
        "Nombre d'éléments avant : {n_elements}, nb_verts : {n_verts_init}, nb elems after {n_m_elements}, nb verts after {n_m_verts}",
    );
    Ok(new_mesh)
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
    let h = 1.0 / 2.0_f64.powf(f64::from(args.num_splits as u32)); //taille éléments

    let mesh_elems = msh.n_elems();
    msh.compute_vertex_to_vertices();
    let vtv = msh.get_vertex_to_vertices().unwrap();
    let n_edges = vtv.n_edges();
    println!("Nombre d'éléments initiaux {mesh_elems}");
    println!("Nombre d'aretes : {n_edges}");
    let (bdy, _) = msh.boundary();
    let _topo = msh.compute_topology();
    let geom = LinearGeometry::<3, Triangle>::new(&msh, bdy).unwrap();
    msh.compute_volumes();
    let file_name = "Initial_Mesh.vtu".to_string();
    let output_path = output_dir.join(&file_name);
    let _ = msh.write_vtk(output_path.to_str().unwrap());
    let mut option = true;
    match args.metric_type.as_str() {
        "iso" => {
            msh.compute_vertex_to_elems();
            let _m: Vec<AnisoMetric3d> = if args.option.as_str() == "true" {
                calculate_op_metric_elems(&msh, true) //Collapse Metric 
            } else {
                option = false;
                calculate_op_metric_elems(&msh, false) // Split Metric 
            };
            let m = calculate_iso_metric(&msh, h);
            // println!("Metrique résultante par sommets : {m:?}");
            // let mut result_metrics = Vec::with_capacity(msh.n_elems() as usize);
            // for g_elem in msh.gelems() {
            //     let chosen_metric = g_elem.implied_metric();
            //     result_metrics.push(chosen_metric);
            // }
            // let implied_metric_vec = msh
            //     .elem_data_to_vertex_data_metric(&result_metrics)
            //     .unwrap();
            // println!("Vecteur des métriques induites : {implied_metric_vec:?}");
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
                            option,
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
                            option,
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
                            option,
                        )?;
                    }
                    "BFSWRPartitionner" => {
                        perform_remeshing_sequential::<
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
                            option,
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
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "MetisKWay",
                            args.n_parts,
                            "iso",
                            option,
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        let msh = perform_remeshing_sequential::<
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
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisR_iso_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            } else {
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
                            option,
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
                            option,
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
                            option,
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
                            option,
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        let msh = perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisKWay>,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "MetisKWay",
                            args.n_parts,
                            "iso",
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisKWay_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        let msh = perform_remeshing::<
                            3,
                            Tetrahedron,
                            IsoMetric<3>,
                            MetisPartitioner<MetisRecursive>,
                            TotoCostEstimator<3, Tetrahedron, IsoMetric<3>>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "MetisRecursive",
                            args.n_parts,
                            "iso",
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisR_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
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
            msh.compute_vertex_to_elems();
            option = false;
            let m = calculate_stretching_metric(&msh);

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
                            option,
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
                            option,
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
                            option,
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
                            option,
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
                            msh,
                            &geom,
                            m,
                            "Nocost",
                            "MetisKWay",
                            args.n_parts,
                            "aniso",
                            option,
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        let msh = perform_remeshing::<
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
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisR_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Partitionneur non valide",
                        )));
                    }
                }
            } else {
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
                            option,
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
                            option,
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
                            option,
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
                            option,
                        )?;
                    }
                    #[cfg(feature = "metis")]
                    "MetisKWay" => {
                        let msh = perform_remeshing::<
                            3,
                            Tetrahedron,
                            AnisoMetric3d,
                            MetisPartitioner<MetisKWay>,
                            TotoCostEstimator<3, Tetrahedron, AnisoMetric3d>,
                        >(
                            msh,
                            &geom,
                            m,
                            "Toto",
                            "MetisKWay",
                            args.n_parts,
                            "aniso",
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisKWay_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
                    }
                    #[cfg(feature = "metis")]
                    "MetisRecursive" => {
                        let msh = perform_remeshing::<
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
                            option,
                        )
                        .unwrap();
                        let file_name = "MetisR_Remeshed.vtu".to_string();
                        let output_path = output_dir.join(&file_name);
                        let _ = msh.write_vtk(output_path.to_str().unwrap());
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
