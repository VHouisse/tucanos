//! Mesh partition example
// use rayon::iter::ParallelIterator;

// use rayon::iter::ParallelIterator;
use std::{path::Path, process::Command, time::Instant};
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

pub fn get_isotropic_metric_circular_fine(
    mesh: &SimplexMesh<2, Triangle>,
) -> Result<Vec<IsoMetric<2>>> {
    let mut iso_metrics: Vec<IsoMetric<2>> = Vec::with_capacity(mesh.n_elems() as usize);

    for g_elem in mesh.gelems() {
        let center_point = g_elem.center();
        let x = center_point[0];
        let y = center_point[1];

        let mut dist_sq = (x - CENTER_X).powi(2) + (y - CENTER_Y).powi(2);
        dist_sq = dist_sq.sqrt();
        if dist_sq <= RADIUS_SQ {
            iso_metrics.push(IsoMetric::<2>::from(H_INSIDE_CIRCLE));
        } else {
            iso_metrics.push(IsoMetric::<2>::from(H_OUTSIDE_CIRCLE));
        }
    }
    Ok(iso_metrics)
}
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
    let m = get_isotropic_metric_circular_fine(&msh).unwrap();
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
            let m: Vec<AnisoMetric2d> = msh
                .verts()
                .map(|v| AnisoMetric2d::from_iso(&IsoMetric::<2>::from(calculate_aniso_metric(v))))
                .collect();

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
