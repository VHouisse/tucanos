use env_logger::Env;
use std::{fs, path::Path, process:: Command};
use tucanos::{
     geometry::LinearGeometry, mesh::{ SimplexMesh, Tetrahedron, Triangle}, metric::IsoMetric, remesher::{ParallelRemesher, ParallelRemesherParams, ParallelRemeshingInfo, RemesherParams}, Result 
};

//use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

// pub fn get_h(mesh: &SimplexMesh<2, Triangle>) -> Result<Vec<f64>> {
//     let h_min: f64 = 0.025;
//     let h_max: f64 = 0.1;

//     // Itérer sur les ÉLÉMENTS du maillage pour calculer une densité par élément
//     let h_values_per_element: Vec<f64> = mesh
//         .gelems() // `gelems()` retourne un itérateur sur les éléments géométriques (GTriangle dans ce cas)
//         .map(|g_elem| { // `g_elem` est un GTriangle
//             let center_point = g_elem.center(); // Obtenir le centre de l'élément

//             let x = center_point[0]; // Coordonnée x du centre
//             let y = center_point[1]; // Coordonnée y du centre

//             let exponent_val = -((x - 0.5).powi(2) + (y - 0.25).powi(2)) / 0.25f64.powi(2);
//             let exp_term = f64::exp(exponent_val);

//             h_min + (h_max - h_min) * (1.0 - exp_term)
//         })
//         .collect();

//     Ok(h_values_per_element)
// }

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
    let mut split : bool = true;
   
    let n_parts = 2;

    let mut mean_cost :  Vec<f64> = Vec::new();

    init_log("info");

    let output_dir = Path::new("remesh_output");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }
   
    let fname = "geom3d_3.mesh";
    let fname = Path::new(fname);

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
        )
    }

    for i in 0..2{
    if i == 1{
        split = false;
    }   
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb(fname.to_str().unwrap()).unwrap();
    let mut  h = get_h_3d_split(&mesh).unwrap();
    if !split{
         h = get_h_3d_collapse(&mesh).unwrap();
    }
    let mut m : Vec<_> = h
                                         .iter()
                                         .map(|h_val| IsoMetric::<3>::from(*h_val))
                                         .collect();
    mesh.add_boundary_faces();
    mesh.compute_topology();
    let (bdy,_) = mesh.boundary();
    let geom = LinearGeometry::<3,Triangle>::new(&mesh, bdy).unwrap();
    // Necessary to compute graph etc 
    mesh.reorder_hilbert();
    mesh.compute_vertex_to_elems(); 
    mesh.compute_volumes();
    mesh.compute_face_to_elems();

    let info : ParallelRemeshingInfo;
    let remesher = ParallelRemesher::new(mesh, tucanos::mesh::PartitionType::Hilbert(n_parts)).unwrap();
    let mut file_name = format!("Partitionned_2_split.vtu");
    if !split{
        file_name = format!("Partitionned_2_Collapse.vtu");
    }
    let mut output_path = output_dir.join(&file_name);
    let _ = remesher.partitionned_mesh().write_vtk(output_path.to_str().unwrap(), None, None);
    (mesh,info, m ) = remesher.remesh(&m, &geom, RemesherParams::default(), &ParallelRemesherParams::default()).unwrap();

    file_name = format!("Stats_Split.txt");
    if !split{
        file_name = format!("Stats_Collapse.txt")
    }
    output_path = output_dir.join(&file_name);
    mean_cost.push(load_costs_and_calculate_average(&output_path).unwrap().unwrap());
    file_name = format!("Remeshed_2_split.vtu");
      if !split{
        file_name = format!("Remeshed_2_Collapse.vtu");
    }
    output_path = output_dir.join(&file_name);
    info.print_summary();
    let _ = mesh.write_vtk(output_path.to_str().unwrap(), None, None);

    }
   
    let file_name = format!("Stats_Smooth.txt");
    let output_path = output_dir.join(&file_name);
    mean_cost.push(load_costs_and_calculate_average(&output_path).unwrap().unwrap());

    for cost in &mean_cost{
        println!("Cout : {:.2e}", cost)
    }
    println!("Collapse coûte  : {:.2e} plus cher que le split", mean_cost[1]/mean_cost[0]);
    println!("Smooth coûte  : {:.2e} plus cher que le split", mean_cost[2]/mean_cost[0]);

    Ok(())
}