//! Mesh partition example
use std::{path::Path, process::Command, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Result,
    mesh::{Mesh, partition::HilbertBallPartitioner},
};
use tucanos::{
    geometry::LinearGeometry,
    mesh::{GElem, Point, SimplexMesh, Tetrahedron, Triangle, test_meshes::test_mesh_3d},
    metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, IsoMetric},
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

fn main() -> Result<()> {
    let fname = "geom3d.mesh";
    let fname = Path::new(fname);

    let output_dir = Path::new("Results_For_Remeshing_3d");
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
    let mut msh = test_mesh_3d().split().split().split();

    let h = |p: Point<3>| {
        const CENTER_X: f64 = 0.3;
        const CENTER_Y: f64 = 0.3;
        const CENTER_Z: f64 = 0.3;
        const RADIUS: f64 = 0.1;
        const RADIUS_SQ_ACTUAL: f64 = RADIUS * RADIUS;
        const H_INSIDE_CIRCLE: f64 = 0.001;
        const H_OUTSIDE_CIRCLE: f64 = 0.1;

        let mut res = H_OUTSIDE_CIRCLE;

        let x = p[0];
        let y = p = p[1];
        let z = p[2];
        let dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2) + (z - CENTER_Z).powi(2);

        if dist_sq <= RADIUS_SQ_ACTUAL {
            res = H_INSIDE_CIRCLE;
        }

        res
    };
    let m = msh
        .verts()
        .map(|v| {
            let metric_value = h(v); // Calculez la valeur scalaire (f64) de la métrique en utilisant ce point
            IsoMetric::<3>::from(metric_value)
        })
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
    >::new(msh, m, 4)?;
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

    Ok(())
}
