//! Mesh partition example
// use rayon::iter::ParallelIterator;

// use rayon::iter::ParallelIterator;
use std::{path::Path, process::Command, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{Result, mesh::Mesh};
use tucanos::{
    geometry::LinearGeometry,
    mesh::{Edge, GElem, SimplexMesh, Triangle, test_meshes::test_mesh_2d},
    metric::IsoMetric,
    remesher::{ParallelRemesher, ParallelRemesherParams, RemesherParams, TotoCostEstimator},
};
/// .geo file to generate the input mesh with gmsh:
const GEO_FILE: &str = r#"// Gmsh project created on Tue Jun 10 20:58:23 2025
SetFactory("OpenCASCADE");
Cone(1) = {0, 0, 0, 1, 0, 0, 0.5, 0.1, 2*Pi};
Sphere(2) = {0, 0, 0, 0.1, -Pi/2, Pi/2, 2*Pi};
BooleanDifference{ Curve{2}; Volume{1}; Delete; }{ Volume{2}; Delete; }
MeshSize {3} = 0.01;
MeshSize {4} = 0.001;

Physical Surface("cone", 12) = {1};
Physical Surface("top", 13) = {2};
Physical Surface("bottom", 14) = {3};
Physical Surface("sphere", 15) = {4, 5};
Physical Volume("E", 16) = {1};

"#;
const CENTER_X: f64 = 0.3;
const CENTER_Y: f64 = 0.3;
const RADIUS_SQ: f64 = 0.1 * 1.0;

const H_INSIDE_CIRCLE: f64 = 0.01;
const H_OUTSIDE_CIRCLE: f64 = 0.5;

fn main() -> Result<()> {
    let fname = "geom3d.mesh";
    let fname = Path::new(fname);

    let output_dir = Path::new("Results_For_Remeshing_2d");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
    if !fname.exists() {
        std::fs::write("geom3d.geo", GEO_FILE)?;

        let output = Command::new("gmsh")
            .arg("geom3d.geo")
            .arg("-3")
            .arg("-o")
            .arg(fname.to_str().unwrap())
            .output()?;

        assert!(
            output.status.success(),
            "gmsh error: {}",
            String::from_utf8(output.stderr).unwrap()
        );
    }
    let mut msh = test_mesh_2d().split().split().split().split().split();
    println!(
        "Nombre d'éléments contenus dans le maillage {} ",
        msh.n_elems()
    );

    let h = |p: Point<2>| {
        let mut res = 0;
        let x = p[0];
        let y = p[1];
        let mut dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);
        dist_sq = dist_sq.sqrt();
        if dist_sq <= RADIUS_SQ {
            res = H_INSIDE_CIRCLE
        } else {
            res = H_OUTSIDE_CIRCLE
        }
        res
    };
    let m = msh
        .verts()
        .map(|v| {
            let metric_value = h(v); // Calculez la valeur scalaire (f64) de la métrique en utilisant ce point
            IsoMetric::<2>::from(metric_value)
        })
        .collect();
    let (bdy, _) = msh.boundary();
    let _topo = msh.compute_topology();
    let geom = LinearGeometry::<2, Edge>::new(&msh, bdy).unwrap();
    msh.compute_volumes();
    // let imp_met: Vec<_> = msh
    //     .par_gelems()
    //     .map(|ge| ge.calculate_implied_metric())
    //     .collect();
    let remesher = ParallelRemesher::<
        2,
        Triangle,
        IsoMetric<2>,
        tmesh::mesh::partition::HilbertBallPartitioner,
        TotoCostEstimator<2, Triangle, IsoMetric<2>>,
    >::new(msh, m, 8)?;
    let file_name = format!("Partitionned_Hilbert.vtu");
    let output_path = output_dir.join(&file_name);
    remesher
        .partitionned_mesh()
        .write_vtk(output_path.to_str().unwrap())?;
    let dd_params = ParallelRemesherParams::default();
    let params = RemesherParams::default();
    let time = Instant::now();
    (msh, _, _) = remesher.remesh(&geom, params, &dd_params).unwrap();
    let t2 = time.elapsed();
    println!("Temps de remaillage Avec Estimation du travail {:?}", t2);
    let file_name = format!("Remeshed_Hilbert.vtu");
    let output_path = output_dir.join(&file_name);
    msh.write_vtk(output_path.to_str().unwrap())?;
    Ok(())
}
