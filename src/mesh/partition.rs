use crate::{
    mesh::{ordering::hilbert_indices, ConnectedComponents, ConnectedComponentsInfo, Elem, GElem, SimplexMesh}, Idx, Result, Tag
};
use log::{debug, warn};
use std::{collections::{HashMap, HashSet}, fmt};
use rustc_hash::FxHashSet;
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum PartitionType {
    Hilbert(Idx),
    Scotch(Idx),
    MetisRecursive(Idx),
    MetisKWay(Idx),
    HilbertB(Idx),
    None,
}
impl fmt::Display for PartitionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PartitionType::Hilbert(n) => write!(f, "Hilbert_{}", n), 
            PartitionType::Scotch(n) => write!(f, "Scotch_{}", n),
            PartitionType::MetisRecursive(n) => write!(f, "MetisRecursive_{}", n),
            PartitionType::MetisKWay(n) => write!(f, "MetisKWay_{}", n),
            PartitionType::HilbertB(n) => write!(f, "Hilbert_Ball_{}", n),
            PartitionType::None => write!(f,"None")
            
        }
    }
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn partition(&mut self, ptype: PartitionType) -> Result<()> {
        match ptype {
            PartitionType::Hilbert(n) => {
                self.partition_hilbert(n);
                Ok(())
            }
            PartitionType::Scotch(n) => self.partition_scotch(n),
            PartitionType::MetisRecursive(n) => self.partition_metis(n, "recursive"),
            PartitionType::MetisKWay(n) => self.partition_metis(n, "kway"),
            PartitionType::HilbertB(n) => {
                self.partition_hilbert_ball( n );
                Ok(())},
            PartitionType::None => unreachable!(),
        }
    }

    pub fn partition_hilbert(&mut self, n_parts: Idx) {
        debug!("Partition the mesh into {n_parts} using a Hilbert curve");

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
        } else {
            let indices = hilbert_indices(self.bounding_box(), self.gelems().map(|ge| ge.center()));

            let m = self.n_elems() / n_parts + 1;
            let partition = indices.iter().map(|&i| (i / m) as Tag).collect::<Vec<_>>();

            self.mut_etags()
                .enumerate()
                .for_each(|(i, t)| *t = partition[i] as Tag + 1);
        }
    }

    pub fn partition_hilbert_ball(&mut self, n_parts: Idx){
        debug!("Partition the mesh into {n_parts} using a Hilbert curve & elmnt ball");
        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }
        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
           
        } else {
                let weights = self.get_elem_work().unwrap();
                // Suppose that elements have been renumbered using Hilbert Curve  
                let total_work : f64 = weights.iter().sum();
                let work_per_partition = total_work / n_parts as f64; 
                let indices : Vec<u32> = (0..self.n_verts() as u32).collect();
    
                let v2e_graph = self.get_vertex_to_elems().unwrap();
                let mut assigned_elements : HashSet<Idx> = HashSet::new();
                
                let mut partition : Vec<Tag> = vec![0;self.n_elems() as usize]; 
                let mut current_partition_idx = 0;
                let mut current_work_partition = 0.0;

                //To parallelize 
                for &vertex_id in indices.iter(){
                    let element_in_ball = v2e_graph.row(vertex_id);
                    for &elem_idx in element_in_ball{
                        if !assigned_elements.contains(&elem_idx){
                            let elem_work = weights[elem_idx as usize];
                            if (current_work_partition + elem_work ) > work_per_partition{
                                current_work_partition = 0.0;
                                if current_partition_idx + 2 <= n_parts{
                                    current_partition_idx +=1;
                                }
                            }else{
                                current_work_partition += elem_work;
                            }
                            partition[elem_idx as usize] = current_partition_idx as Tag;
                            assigned_elements.insert(elem_idx);
                        }
                    }
                }

                self.mut_etags()
                .enumerate()
                .for_each(|(i, t)| *t = partition[i] as Tag + 1);
        }
        
    }

    // Depending on the type of connectivity used 
    pub fn partition_correction(&mut self, n_parts: Idx){        
        debug!("Correcting Connected Components"); 
        let work = self.get_elem_work().unwrap();
        let mut cc_infos: Vec<ConnectedComponentsInfo> = Vec::new();
        for i_part in 0..n_parts{
            let mut smsh = self.extract_tag(i_part as Tag + 1);
            let e2e = smsh.mesh.compute_elem_to_elems(); 
            let cc = ConnectedComponents::<Idx>::new(e2e);
            let mut _n_cc = 1; 
            if let Ok(cc_graph) = cc{
                _n_cc = cc_graph.tags().iter().clone().collect::<FxHashSet<_>>().len();
                let cc_tags = cc_graph.tags();
                // Regroupe les éléments d'une composante connexe d'un subMeshpar un id
                let mut current_partition_cc : HashMap<Idx, Vec<Idx>> = HashMap::new();
                for(sub_elem_idx, &cc_id) in cc_tags.iter().enumerate(){
                    let parent_elemnt_id  = smsh.parent_elem_ids[sub_elem_idx];
                    current_partition_cc.entry(cc_id).or_default().push(parent_elemnt_id);
                }

                let mut current_partition_cc_infos: Vec<ConnectedComponentsInfo> = Vec::new();
                let mut max_work_per_cc = 0.0 ; 
                let mut _primary_cc_id = Idx::MAX;

                for (cc_id, elements_in_cc) in current_partition_cc{
                    let current_cc_work : f64 = elements_in_cc
                                                .iter()
                                                .map(|&elemn_parent_id| work[elemn_parent_id as usize])
                                                .sum();
                    current_partition_cc_infos.push(ConnectedComponentsInfo{
                        cc_idx : cc_id,
                        elements : elements_in_cc,
                        total_work : current_cc_work,
                        is_primary : false,
                        partition_id : i_part as Tag + 1

                    });
                    
                    if current_cc_work > max_work_per_cc {
                        max_work_per_cc  = current_cc_work;
                       _primary_cc_id = cc_id;
                    }

                    for cc_info in &mut current_partition_cc_infos {
                        if cc_info.cc_idx == _primary_cc_id {
                            cc_info.is_primary = true;
                        }
                    }
                    
                }
                cc_infos.extend(current_partition_cc_infos);
    
            }

        }

        // Fusion Part

      
    }


    #[allow(clippy::needless_pass_by_ref_mut)]
    #[cfg(not(feature = "scotch"))]
    pub fn partition_scotch(&mut self, _n_parts: Idx) -> Result<()> {
        use crate::Error;
        Err(Error::from("the scotch feature is not enabled"))
    }

    /// Partition the mesh using scotch into `n_parts`. The partition id, defined for all the elements
    /// is stored in self.etags
    #[cfg(feature = "scotch")]
    pub fn partition_scotch(&mut self, n_parts: Idx) -> Result<()> {
        debug!("Partition the mesh into {n_parts} using scotch");

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
            return Ok(());
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.compute_elem_to_elems();

        let architecture = scotch::Architecture::complete(n_parts as i32);

        let xadj: Vec<scotch::Num> = e2e
            .ptr
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();
        let adjncy: Vec<scotch::Num> = e2e
            .indices
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();

        let mut graph = scotch::Graph::build(&scotch::graph::Data::new(
            0,
            &xadj,
            &[],
            &[],
            &[],
            &adjncy,
            &[],
        ))
        .unwrap();
        graph.check().unwrap();
        graph
            .mapping(&architecture, &mut partition)
            .compute(&mut scotch::Strategy::new())?;

        self.mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = partition[i] as Tag + 1);

        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[cfg(not(feature = "metis"))]
    pub fn partition_metis(&mut self, _n_parts: Idx, _method: &str) -> Result<()> {
        use crate::Error;
        Err(Error::from("the metis feature is not enabled"))
    }

    /// Partition the mesh using metis into `n_parts`. The partition id, defined for all the elements
    /// is stored in self.etags
    #[cfg(feature = "metis")]
    pub fn partition_metis(&mut self, n_parts: Idx, method: &str) -> Result<()> {
        debug!("Partition the mesh into {} using metis", n_parts);

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
            return Ok(());
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.compute_elem_to_elems();

        let mut xadj: Vec<metis::Idx> = e2e
            .ptr
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();
        let mut adjncy: Vec<metis::Idx> = e2e
            .indices
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();
        
        let graph = metis::Graph::new(1, n_parts as metis::Idx, &mut xadj, &mut adjncy);
        match method {
            "recursive" => graph.part_recursive(&mut partition).unwrap(),
            "kway" => graph.part_kway(&mut partition).unwrap(),
            _ => unreachable!("Unknown method"),
        };

        self.mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = partition[i] as Tag + 1);

        Ok(())
    }

    /// Get the partition quality (ration of the number of interface faces to the total number of faces)
    pub fn partition_quality(&self) -> Result<f64> {
        let f2e = self.get_face_to_elems()?;

        let n = f2e
            .iter()
            .filter(|(_, v)| v.len() == 2 && self.etag(v[0]) != self.etag(v[1]))
            .count();
        Ok(n as f64 / f2e.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Result,
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
    };

    #[test]
    fn test_partition_hilbert_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_hilbert(4);

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[test]
    fn test_partition_hilbert_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_hilbert(4);

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "recursive")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "recursive")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d_kway() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "kway")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d_kway() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "kway")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.022, "failed, q = {q}");

        Ok(())
    }
}
