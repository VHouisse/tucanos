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

const ANISO_CENTER_X: f64 = 0.5;
const ANISO_CENTER_Y: f64 = 0.25;
const ANISO_CENTER_Z: f64 = 0.5; 
const ANISO_RADIUS_SQ: f64 = 0.25 * 0.2; 
const H_ANISO_COARSE: f64 = 0.1; 
const H_BACKGROUND_ISO: f64 = 0.05; 


pub fn get_aniso_metric_x_2d(mesh: &SimplexMesh<2, Triangle>) -> Result<Vec<AnisoMetric2d>> {
    let mut aniso_metrics: Vec<AnisoMetric2d> = Vec::with_capacity(mesh.n_elems() as usize);

    for g_elem in mesh.gelems() {
        let center_point = g_elem.center();
        let x = center_point[0];
        let y = center_point[1];

        let dist_sq = (x - ANISO_CENTER_X).powi(2) + (y - ANISO_CENTER_Y).powi(2);

        if dist_sq < ANISO_RADIUS_SQ {
           
            let s0 = Point::<2>::new(H_ANISO_FINE, 0.0); 
            let s1 = Point::<2>::new(0.0, H_ANISO_COARSE); 
            aniso_metrics.push(AnisoMetric2d::from_sizes(&s0, &s1));
        } else {
            
            aniso_metrics.push(AnisoMetric2d::from_iso(&IsoMetric::<2>::from(
                H_BACKGROUND_ISO,
            )));
        }
    }

    Ok(aniso_metrics)
}

pub fn get_aniso_metric_x_3d(mesh: &SimplexMesh<3, Tetrahedron>) -> Result<Vec<AnisoMetric3d>> {
    let mut aniso_metrics: Vec<AnisoMetric3d> = Vec::with_capacity(mesh.n_elems() as usize);

    for g_elem in mesh.gelems() {
        let center_point = g_elem.center();
        let x = center_point[0];
        let y = center_point[1];
        let z = center_point[2];

        // Calculer la distance au centre de la zone anisotrope au carré
        let dist_sq = (x - ANISO_CENTER_X).powi(2)
            + (y - ANISO_CENTER_Y).powi(2)
            + (z - ANISO_CENTER_Z).powi(2);

        if dist_sq < ANISO_RADIUS_SQ {
            // À l'intérieur de la zone d'intérêt : métrique anisotrope
            // Très fine en X (h_x = H_ANISO_FINE)
            // Plus grossière en Y (h_y = H_ANISO_COARSE) et Z (h_z = H_ANISO_COARSE)
            let s0 = Point::<3>::new(H_ANISO_FINE, 0.0, 0.0); // Axe X
            let s1 = Point::<3>::new(0.0, H_ANISO_COARSE, 0.0); // Axe Y
            let s2 = Point::<3>::new(0.0, 0.0, H_ANISO_COARSE); // Axe Z
            aniso_metrics.push(AnisoMetric3d::from_sizes(&s0, &s1, &s2));
        } else {
            // En dehors de la zone : métrique isotrope de fond
            aniso_metrics.push(AnisoMetric3d::from_iso(&IsoMetric::<3>::from(
                H_BACKGROUND_ISO,
            )));
        }
    }

    Ok(aniso_metrics)
}
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
    let m = get_aniso_metric_x_3d(&msh).unwrap();
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

    let dd_params = ParallelRemesherParams::default();
    let params = RemesherParams::default();
    let time = Instant::now();
    (msh, _, _) = remesher.remesh(&geom, params, &dd_params).unwrap();
    let t2 = time.elapsed();
    println!("Temps de remaillage Avec Estimation du travail {:?}", t2);
    let file_name = format!("Partitionned_Hilbert.vtu");
    let output_path = output_dir.join(&file_name);
    msh.write_vtk(output_path.to_str().unwrap())?;

    Ok(())
}
