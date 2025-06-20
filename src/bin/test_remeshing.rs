use env_logger::Env;
use std::{intrinsics::exp2f32, path::Path};
use tucanos::{
    geometry::LinearGeometry, mesh::{test_meshes::test_mesh_2d, SimplexMesh, Tetrahedron, Triangle}, metric::{IsoMetric}, Result, Tag
};
use tucanos::mesh::Point;
use tucanos::remesher::{Remesher, RemesherParams};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}


pub fn get_h(mesh: &SimplexMesh<2, Triangle>) -> Result<Vec<f64>> {
    let h_min: f64 = 0.001;
    let h_max: f64 = 0.05;

    let h_values: Vec<f64> = mesh
        .verts()
        .map(|p| {
            let x = p[0]; 
            let y = p[1]; 

            let exponent_val = -((x - 0.5).powi(2) + (y - 0.25).powi(2)) / 0.25f64.powi(2);
            let exp_term = f64::exp(exponent_val);

            h_min + (h_max - h_min) * (1.0 - exp_term)
        })
        .collect(); 

    Ok(h_values)
}

pub fn test_remeshing() -> Result<()> {
    init_log("info");

    let output_dir = Path::new("remesh_output");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)?;
    }

    let mut mesh = test_mesh_2d().split().split().split();
    let n = mesh.n_elems() as usize;
    let h = get_h(&mesh).unwrap();
    
    let iso_metric : Vec<_> = h
                                         .iter()
                                         .map(|h_val| IsoMetric::<2>::from(*h_val))
                                         .collect();
    mesh.add_boundary_faces();
    mesh.compute_topology();
    let (bdy,_) = mesh.boundary();
    let geom = LinearGeometry::<2,Triangle>::new(&mesh, bdy);

    let params = RemesherParams {
        debug: false,
        ..RemesherParams::default()
    };

    let topo = mesh.get_topology()?;
    let mut remesher = Remesher::new(&mesh, &iso_metric, &geom)?;

    remesher.remesh(&params, &default_geom)?;
    let remeshed_mesh = remesher.to_mesh(true);

    let file_name = format!("cube_remeshed.vtu");
    let output_path = output_dir.join(&file_name);
    remeshed_mesh.write_vtk(output_path.to_str().unwrap(), None, None)?;

    println!("Mesh du cube remaillé et sauvegardé dans {:?}", output_path);

    Ok(())
}