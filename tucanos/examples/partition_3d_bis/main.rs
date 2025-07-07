//! Mesh partition example
use env_logger::Env;
use std::{fs, path::Path, process::Command, time::Instant};
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
    mesh::{GElem, SimplexMesh, Tetrahedron, Triangle, test_meshes::test_mesh_3d},
    metric::IsoMetric,
    remesher::{ElementCostEstimator, TotoCostEstimator},
};
//use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

pub fn get_h(mesh: &SimplexMesh<2, Triangle>) -> Result<Vec<f64>> {
    let h_min: f64 = 0.025;
    let h_max: f64 = 0.1;

    // Itérer sur les ÉLÉMENTS du maillage pour calculer une densité par élément
    let h_values_per_element: Vec<f64> = mesh
        .gelems() // `gelems()` retourne un itérateur sur les éléments géométriques (GTriangle dans ce cas)
        .map(|g_elem| {
            // `g_elem` est un GTriangle
            let center_point = g_elem.center(); // Obtenir le centre de l'élément

            let x = center_point[0]; // Coordonnée x du centre
            let y = center_point[1]; // Coordonnée y du centre

            let exponent_val = -((x - 0.5).powi(2) + (y - 0.25).powi(2)) / 0.25f64.powi(2);
            let exp_term = f64::exp(exponent_val);

            h_min + (h_max - h_min) * (1.0 - exp_term)
        })
        .collect();

    Ok(h_values_per_element)
}

pub fn get_h_3d_split(mesh: &SimplexMesh<3, Tetrahedron>) -> Result<Vec<f64>> {
    let h_uniform_split: f64 = 0.025; // Une valeur uniforme très petite pour forcer le raffinement

    let n_elems = mesh.n_elems() as usize; // Nombre d'éléments du maillage

    // Crée un vecteur de `h_uniform_split` pour chaque élément
    let h_values_per_element: Vec<f64> = vec![h_uniform_split; n_elems];

    Ok(h_values_per_element)
}

pub fn get_h_3d_collapse(mesh: &SimplexMesh<3, Tetrahedron>) -> Result<Vec<f64>> {
    // Une valeur uniforme très grande pour forcer le déraffinement
    // Doit être significativement plus grande que les tailles d'éléments du maillage initial.
    let h_uniform_collapse: f64 = 0.1; // Par exemple, 1.0 (ou 10.0)

    let n_elems = mesh.n_elems() as usize; // Nombre d'éléments du maillage

    // Crée un vecteur de `h_uniform_collapse` pour chaque élément
    let h_values_per_element: Vec<f64> = vec![h_uniform_collapse; n_elems];

    Ok(h_values_per_element)
}

pub fn load_costs_and_calculate_average(file_path: &Path) -> Result<Option<f64>> {
    if !file_path.exists() {
        return Ok(None); // Fichier non trouvé
    }

    let content = fs::read_to_string(file_path)?;
    let costs: Vec<f64> = content
        .lines()
        .filter_map(|line| line.parse::<f64>().ok()) // Tente de parser chaque ligne en f64
        .collect();

    if costs.is_empty() {
        Ok(None) // Le fichier est vide ou ne contient pas de nombres valides
    } else {
        let sum_costs: f64 = costs.iter().sum();
        let average_cost = sum_costs / costs.len() as f64;
        Ok(Some(average_cost))
    }
}

fn print_partition_cc(msh: &SimplexMesh<3, Tetrahedron>, n_parts: usize) {
    for i in 0..n_parts {
        let pmesh = msh.get_partition(i).mesh;
        let faces = pmesh.all_faces();
        let graph = pmesh.element_pairs(&faces);
        let cc = graph.connected_components().unwrap();
        let n_cc = cc.iter().copied().max().unwrap_or(0) + 1;
        println!("  part {i}: {n_cc} components");
    }
}

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

pub fn main() -> Result<()> {
    let fname = "geom3d.mesh";
    let fname = Path::new(fname);

    let output_dir = Path::new("Global_Partitionnement");
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

    println!("# of elements: {}", msh.n_elems());

    let n_parts = 4;
    let h = get_h_3d_split(&msh).unwrap();
    let m: Vec<_> = h.iter().map(|h_val| IsoMetric::<3>::from(*h_val)).collect();
    msh.compute_volumes();
    let estimator = TotoCostEstimator::<3, Tetrahedron, IsoMetric<3>>::new();
    let weights = estimator.compute(&msh, &m);
    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertPartitioner>(n_parts, Some(weights))?;
    let file_name = format!("Partitionned_Hilbert.vtu");
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
    let file_name = format!("Partitionned_Hilbert_Ball.vtu");
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
    let file_name = format!("Partitionned_BFS.vtu");
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
    let file_name = format!("Partitionned_BFSWR.vtu");
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

    // let start = Instant::now();
    // let (quality, imbalance) = msh.partition::<RCMPartitioner>(n_parts, None)?;
    // let t = start.elapsed();
    // println!(
    //     "RCMPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
    //     t.as_secs_f64(),
    //     quality,
    //     imbalance
    // );
    // print_partition_cc(&msh, n_parts);

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
