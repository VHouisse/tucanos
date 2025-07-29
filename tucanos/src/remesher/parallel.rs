use super::ElementCostEstimator;

use crate::remesher::stats::StepStats;
use crate::{
    Idx, Result, Tag,
    geometry::Geometry,
    mesh::{Elem, HasTmeshImpl, SimplexMesh, SubSimplexMesh},
    metric::{HasImpliedMetric, IsoMetric, Metric},
    remesher::{Remesher, RemesherParams},
};
use log::{debug, warn};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;
use serde::Serialize;
use std::{marker::PhantomData, sync::Mutex, time::Instant};
use tmesh::mesh::partition::Partitioner;

#[derive(Clone, Debug)]
pub struct ParallelRemesherParams {
    pub n_layers: Idx,
    pub level: Idx,
    pub max_levels: Idx,
    pub min_verts: Idx,
}

impl ParallelRemesherParams {
    #[must_use]
    pub const fn new(n_layers: Idx, max_levels: Idx, min_verts: Idx) -> Self {
        Self {
            n_layers,
            level: 0,
            max_levels,
            min_verts,
        }
    }

    #[must_use]
    pub const fn n_layers(&self) -> Idx {
        self.n_layers
    }

    #[must_use]
    pub const fn level(&self) -> Idx {
        self.level
    }

    #[must_use]
    const fn next(&self, n_verts: Idx) -> Option<Self> {
        if self.level + 1 < self.max_levels && n_verts > self.min_verts {
            Some(Self {
                n_layers: self.n_layers,
                level: self.level + 1,
                max_levels: self.max_levels,
                min_verts: self.min_verts,
            })
        } else {
            None
        }
    }
}

impl Default for ParallelRemesherParams {
    fn default() -> Self {
        Self::new(2, 1, 10000)
    }
}

#[derive(Default, Clone, Serialize)]
pub struct RemeshingInfo {
    pub n_verts_init: Idx,
    pub n_verts_final: Idx,
    pub time: f64,
    pub remesh_stats: Vec<StepStats>,
}
impl RemeshingInfo {
    #[allow(clippy::too_many_lines)]
    fn print_summary_remesh_stats(&self) {
        let mut nb_split_step = 0;
        let mut total_verif_splits = 0;
        let mut total_splits = 0;
        let mut total_splits_time_absolute = 0.0;
        let mut total_splits_fails = 0;
        let mut total_split_fail_time_absolute = 0.0;
        let mut total_verifs_time_splits_absolute = 0.0;

        let mut nb_collapse_step = 0;
        let mut total_verif_collapses = 0;
        let mut total_collapses = 0;
        let mut total_collapses_fails = 0;
        let mut total_collapse_success_time_absolute = 0.0;
        let mut total_collapse_fail_time_absolute = 0.0;
        let mut total_verifs_time_collapses_absolute = 0.0;

        let mut total_swaps_performed = 0;
        let mut total_swaps_fails = 0;
        let mut total_swaps_verifs = 0;
        let mut total_swap_success_time_absolute = 0.0;
        let mut total_swap_fail_time_absolute = 0.0;
        let mut total_swap_verif_time_absolute = 0.0;

        let mut total_smooth_fails = 0;
        let mut _t_init = 0;

        for step_stat in self.remesh_stats.clone() {
            match step_stat {
                StepStats::Split(s) => {
                    nb_split_step += 1;
                    total_verif_splits += s.get_n_verifs();
                    total_splits += s.get_n_splits();
                    total_splits_fails += s.get_n_fails();
                    total_split_fail_time_absolute += s.get_t_time_fails();
                    total_splits_time_absolute += s.get_t_time_split();
                    total_verifs_time_splits_absolute += s.get_t_time_verif();
                }
                StepStats::Collapse(s) => {
                    nb_collapse_step += 1;
                    total_collapses += s.get_n_collapses();
                    total_collapses_fails += s.get_n_fails();
                    total_verif_collapses += s.get_n_verifs();
                    total_collapse_success_time_absolute += s.get_t_time_collapse();
                    total_collapse_fail_time_absolute += s.get_t_time_fails();
                    total_verifs_time_collapses_absolute += s.get_t_time_verif();
                }

                StepStats::Swap(s) => {
                    total_swaps_performed += s.get_n_swaps();
                    total_swaps_fails += s.get_n_fails();
                    total_swaps_verifs += s.get_n_verifs();
                    total_swap_success_time_absolute += s.get_t_time_swaps_success();
                    total_swap_fail_time_absolute += s.get_t_time_swaps_fails();
                    total_swap_verif_time_absolute += s.get_t_time_swaps_verif();
                }
                StepStats::Smooth(s) => {
                    total_smooth_fails += s.get_n_fails();
                }
                StepStats::Init(_s) => {
                    _t_init = 0;
                }
            }
        }

        println!("\n--- Remeshing Summary Stats ---");

        println!(
            "\nSplits (Total in Remeshing Process):
          Total Split Steps: {nb_split_step}
          Total Splits Performed: {total_splits}
          Total Failed Splits: {total_splits_fails}
          Total Verifications for Splits: {total_verif_splits}

          Overall Average Time per Succeeded Split: {:.2e} s
          Total Time for All Succeeded Splits: {:.2e} s

          Overall Average Time per Failed Split: {:.2e} s
          Total Time for All Failed Splits: {:.2e} s

          Overall Average Time per Split Verification: {:.2e} s
          Total Time for All Split Verifications: {:.2e} s",
            if total_splits > 0 {
                total_splits_time_absolute / f64::from(total_splits)
            } else {
                0.0
            },
            total_splits_time_absolute,
            if total_splits_fails > 0 {
                total_split_fail_time_absolute / f64::from(total_splits_fails)
            } else {
                0.0
            },
            total_split_fail_time_absolute,
            if total_verif_splits > 0 {
                total_verifs_time_splits_absolute / f64::from(total_verif_splits)
            } else {
                0.0
            },
            total_verifs_time_splits_absolute,
        );

        println!(
            "\nCollapses (Total in Remeshing Process):
          Total Collapse Steps: {nb_collapse_step}
          Total Collapses Performed: {total_collapses}
          Total Failed Collapses: {total_collapses_fails}
          Total Verifications for Collapses: {total_verif_collapses}

          Overall Average Time per Succeeded Collapse: {:.2e} s
          Total Time for All Succeeded Collapses: {:.2e} s

          Overall Average Time per Failed Collapse: {:.2e} s
          Total Time for All Failed Collapses: {:.2e} s

          Overall Average Time per Collapse Verification: {:.2e} s
          Total Time for All Collapse Verifications: {:.2e} s",
            if total_collapses > 0 {
                total_collapse_success_time_absolute / f64::from(total_collapses)
            } else {
                0.0
            },
            total_collapse_success_time_absolute,
            if total_collapses_fails > 0 {
                total_collapse_fail_time_absolute / f64::from(total_collapses_fails)
            } else {
                0.0
            },
            total_collapse_fail_time_absolute,
            if total_verif_collapses > 0 {
                total_verifs_time_collapses_absolute / f64::from(total_verif_collapses)
            } else {
                0.0
            },
            total_verifs_time_collapses_absolute,
        );

        println!(
            "\nSwaps (Total in Remeshing Process):
          Total Swaps Performed: {total_swaps_performed}
          Total Failed Swaps: {total_swaps_fails}
          Total Verifications for Swaps: {total_swaps_verifs}

          Overall Average Time per Succeeded Swap: {:.2e} s
          Total Time for All Succeeded Swaps: {:.2e} s

          Overall Average Time per Failed Swap: {:.2e} s
          Total Time for All Failed Swaps: {:.2e} s

          Overall Average Time per Swap Verification: {:.2e} s
          Total Time for All Swap Verifications: {:.2e} s",
            // Calculation for number of successful swaps
            if total_swaps_performed - total_swaps_fails - total_swaps_verifs > 0 {
                total_swap_success_time_absolute
                    / f64::from(total_swaps_performed - total_swaps_fails - total_swaps_verifs)
            } else {
                0.0
            },
            total_swap_success_time_absolute,
            if total_swaps_fails > 0 {
                total_swap_fail_time_absolute / f64::from(total_swaps_fails)
            } else {
                0.0
            },
            total_swap_fail_time_absolute,
            if total_swaps_verifs > 0 {
                total_swap_verif_time_absolute / f64::from(total_swaps_verifs)
            } else {
                0.0
            },
            total_swap_verif_time_absolute,
        );

        println!(
            "\nSmooth Operations (Total in Remeshing Process):
          Total Failed Smooths: {total_smooth_fails}",
        );

        println!("\n------------------------------------");
    }
}

#[derive(Default, Serialize)]
pub struct ParallelRemeshingInfo {
    info: RemeshingInfo,
    partition_time: f64,
    partition_quality: f64,
    partition_imbalance: f64,
    partitions: Vec<RemeshingInfo>,
    interface: Option<Box<ParallelRemeshingInfo>>,
}

impl ParallelRemeshingInfo {
    fn print_short(&self, indent: String) {
        let s = &self.info;
        if self.partition_quality > 0.0 {
            println!(
                "{} -> {} verts, partition quality = {}, partition imbalance = {}, {:.2e} secs",
                s.n_verts_init,
                s.n_verts_final,
                self.partition_quality,
                self.partition_imbalance,
                s.time,
            );
            for (i, s) in self.partitions.iter().enumerate() {
                println!(
                    "{indent} partition {i}: {} -> {} verts, {:.2e} secs",
                    s.n_verts_init, s.n_verts_final, s.time
                );
                s.print_summary_remesh_stats();
            }
            if let Some(ifc) = &self.interface {
                print!("{indent} interface: ");
                ifc.print_short(indent + "  ");
            }
        } else {
            println!(
                "{} -> {} verts, {:.2e} secs",
                s.n_verts_init, s.n_verts_final, s.time,
            );
        }
    }

    pub fn print_summary(&self) {
        self.print_short(String::from("  "));
    }

    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}

/// Domain decomposition
pub struct ParallelRemesher<
    const D: usize,
    E: Elem,
    M: Metric<D>,
    P: Partitioner,
    C: ElementCostEstimator<D, E, M>,
> {
    mesh: SimplexMesh<D, E>,
    metric: Vec<M>,
    n_parts: Idx,
    partition_tags: Vec<Tag>,
    partition_bdy_tags: Vec<Tag>,
    interface_bdy_tag: Tag,
    partition_time: f64,
    partition_quality: f64,
    partition_imbalance: f64,
    debug: bool,
    _partitioner: PhantomData<P>,
    _cost_estimator: PhantomData<C>,
}

impl<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
    P: Partitioner,
    C: ElementCostEstimator<D, E, M>,
> ParallelRemesher<D, E, M, P, C>
where
    SimplexMesh<D, E>: HasTmeshImpl<D, E>,
    SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face>,
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    /// Create a new parallel remesher based on domain decomposition.
    /// If part is `PartitionType::Scotch(n)` or `PartitionType::Metis(n)` the mesh is partitionned into n subdomains using
    /// scotch / metis. If None, the element tag in `mesh` is used as the partition Id
    ///
    /// NB: the mesh element tags will be modified
    pub fn new(mut mesh: SimplexMesh<D, E>, metric: Vec<M>, n_parts: Idx) -> Result<Self> {
        assert_eq!(mesh.n_verts() as usize, metric.len());
        // Partition
        // Renumbering based on Hilbert
        let now = Instant::now();
        let estimator = C::new(&metric);
        let weights = estimator.compute(&mesh, &metric); //println!("Weights {weights:?}");
        let (partition_quality, partition_imbalance) =
            mesh.partition_elems::<P>(n_parts, Some(weights))?;
        println!(
            "Qualité de la partition : {partition_quality}, Répartition du travail : {partition_imbalance}"
        );
        let partition_time = now.elapsed().as_secs_f64();

        // Get the partition interfaces
        let (bdy_tags, ifc_tags) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());

        let partition_tags = mesh.etags().collect::<FxHashSet<_>>();
        let partition_tags = partition_tags.iter().copied().collect::<Vec<_>>();
        let partition_bdy_tags = ifc_tags.keys().copied().collect::<Vec<_>>();
        debug!("Partition tags: {partition_tags:?}");

        // Use negative tags for interfaces
        mesh.mut_ftags().for_each(|t| {
            if partition_bdy_tags.contains(t) {
                *t = -*t;
            }
        });

        Ok(Self {
            mesh,
            metric,
            n_parts,
            partition_tags,
            partition_bdy_tags: ifc_tags.keys().copied().collect::<Vec<_>>(),
            interface_bdy_tag: Tag::MIN,
            partition_time,
            partition_quality,
            partition_imbalance,
            debug: false,
            _partitioner: PhantomData,
            _cost_estimator: PhantomData,
        })
    }

    pub const fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    #[must_use]
    pub const fn partitionned_mesh(&self) -> &SimplexMesh<D, E> {
        &self.mesh
    }

    #[must_use]
    pub const fn n_verts(&self) -> Idx {
        self.mesh.n_verts()
    }

    /// Get a parallel iterator over the partitiona as SubSimplexMeshes
    #[must_use]
    pub fn par_partitions(&self) -> impl IndexedParallelIterator<Item = SubSimplexMesh<D, E>> + '_ {
        self.partition_tags
            .par_iter()
            .map(|&t| self.mesh.extract_tag(t))
    }

    /// Get an iterator over the partitiona as SubSimplexMeshes
    pub fn seq_partitions(&self) -> impl Iterator<Item = SubSimplexMesh<D, E>> + '_ {
        self.partition_tags
            .iter()
            .map(|&t| self.mesh.extract_tag(t))
    }

    /// Get an element tag that is 2 for the cells that are neighbors of level `n_layers` of the partition interface
    /// (i.e. the faces with a <0 tag)
    #[must_use]
    pub fn flag_interface(&self, mesh: &SimplexMesh<D, E>, n_layers: Idx) -> Vec<Tag> {
        let mut new_etag = vec![1; mesh.n_elems() as usize];

        let mut flag = vec![false; mesh.n_verts() as usize];
        mesh.faces()
            .zip(mesh.ftags())
            .filter(|(_, t)| self.is_partition_bdy(*t))
            .flat_map(|(f, _)| f)
            .for_each(|i| flag[i as usize] = true);

        for _ in 0..n_layers {
            mesh.elems().zip(new_etag.iter_mut()).for_each(|(e, t)| {
                if e.iter().any(|&i_vert| flag[i_vert as usize]) {
                    *t = 2;
                }
            });
            mesh.elems()
                .zip(new_etag.iter())
                .filter(|(_, t)| **t == 2)
                .flat_map(|(e, _)| e)
                .for_each(|i_vert| flag[i_vert as usize] = true);
        }

        new_etag
    }

    fn is_partition_bdy(&self, tag: Tag) -> bool {
        self.partition_bdy_tags.iter().any(|&x| -x == tag)
    }

    const fn is_interface_bdy(&self, tag: Tag) -> bool {
        tag == self.interface_bdy_tag
    }

    fn remesh_submesh<G: Geometry<D>>(
        &self,
        m: &[M],
        geom: &G,
        params: &RemesherParams,
        submesh: SubSimplexMesh<D, E>,
    ) -> (SimplexMesh<D, E>, Vec<M>, Vec<StepStats>) {
        let mut local_mesh = submesh.mesh;

        // to be consistent with the base topology
        local_mesh.mut_etags().for_each(|t| *t = 1);
        let mut topo = self.mesh.get_topology().unwrap().clone();
        topo.clear(|(_, t)| self.is_partition_bdy(t));
        local_mesh.compute_topology_from(topo);

        let local_m: Vec<_> = submesh
            .parent_vert_ids
            .iter()
            .map(|&i| m[i as usize])
            .collect();
        let mut local_remesher = Remesher::new(&local_mesh, &local_m, geom).unwrap();

        let stats = local_remesher.remesh(params, geom).unwrap();

        (
            local_remesher.to_mesh(true),
            local_remesher.metrics(),
            stats,
        )
    }

    /// Remesh using domain decomposition
    #[allow(clippy::too_many_lines)]
    pub fn remesh<G: Geometry<D>>(
        &self,
        geom: &G,
        params: RemesherParams,
        dd_params: &ParallelRemesherParams,
    ) -> Result<(SimplexMesh<D, E>, ParallelRemeshingInfo, Vec<M>)> {
        let res = Mutex::new(SimplexMesh::empty());
        let res_m = Mutex::new(Vec::new());
        let ifc = Mutex::new(SimplexMesh::empty());
        let ifc_m = Mutex::new(Vec::new());

        let level = dd_params.level();

        let info = ParallelRemeshingInfo {
            info: RemeshingInfo {
                n_verts_init: self.mesh.n_verts(),
                ..Default::default()
            },
            partition_time: self.partition_time,
            partition_quality: self.partition_quality,
            partition_imbalance: self.partition_imbalance,
            partitions: vec![RemeshingInfo::default(); self.partition_tags.len()],
            ..Default::default()
        };
        let info = Mutex::new(info);

        if self.debug {
            let fname = format!("level_{level}_init.vtu");
            self.mesh.vtu_writer().export(&fname)?;
        }

        let now = Instant::now();

        self.par_partitions()
            .enumerate()
            .for_each(|(i_part, submesh)| {
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}.vtu");
                    submesh.mesh.vtu_writer().export(&fname).unwrap();
                }

                // Remesh the partition
                debug!("Remeshing level {level} / partition {i_part}");
                let n_verts_init = submesh.mesh.n_verts();
                let now = Instant::now();
                let (mut local_mesh, local_m, stats) =
                    self.remesh_submesh(&self.metric, geom, &params.clone(), submesh);
                // Get the info
                let mut info = info.lock().unwrap();
                info.partitions[i_part] = RemeshingInfo {
                    n_verts_init,
                    n_verts_final: local_mesh.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                    remesh_stats: stats,
                };
                info.print_summary();
                drop(info);

                // Flag elements with n_layers of the interfaces with tag 2, other with tag 1
                let new_etags = self.flag_interface(&local_mesh, dd_params.n_layers);
                local_mesh
                    .mut_etags()
                    .zip(new_etags.iter())
                    .for_each(|(t0, t1)| *t0 = *t1);
                let (bdy_tags, interface_tags) = local_mesh.add_boundary_faces();
                assert!(bdy_tags.is_empty());

                // Flag the faces between elements tagged 1 and 2 as self.interface_bdy_tag
                if interface_tags.is_empty() {
                    warn!("All the elements are in the interface");
                } else {
                    assert_eq!(interface_tags.len(), 1);
                    let tag = interface_tags.keys().next().unwrap();
                    local_mesh.mut_ftags().for_each(|t| {
                        if *t == *tag {
                            *t = self.interface_bdy_tag;
                        }
                    });
                }

                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_remeshed.vtu");
                    local_mesh.vtu_writer().export(&fname).unwrap();
                }

                // Update res
                let mut res = res.lock().unwrap();
                let (ids, _, _) = res.add(&local_mesh, |t| t == 1, |_| true, Some(1e-12));
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_res.vtu");
                    res.vtu_writer().export(&fname).unwrap();
                }
                drop(res);
                let mut res_m = res_m.lock().unwrap();
                res_m.extend(ids.iter().map(|&i| local_m[i]));
                drop(res_m);

                // Update ifc
                let part_tag = 2 + i_part as Tag;
                local_mesh.mut_etags().for_each(|t| {
                    if *t == 2 {
                        *t = part_tag;
                    }
                });
                let mut ifc = ifc.lock().unwrap();
                let (ids, _, _) = ifc.add(&local_mesh, |t| t == part_tag, |_t| true, Some(1e-12));
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_ifc.vtu");
                    ifc.vtu_writer().export(&fname).unwrap();
                }
                drop(ifc);
                let mut ifc_m = ifc_m.lock().unwrap();
                ifc_m.extend(ids.iter().map(|&i| local_m[i]));
            });

        let mut ifc = ifc.into_inner().unwrap();
        if self.debug {
            let fname = format!("level_{level}_ifc.vtu");
            ifc.vtu_writer().export(&fname).unwrap();
            let fname = format!("level_{level}_ifc_bdy.vtu");
            ifc.boundary().0.vtu_writer().export(&fname).unwrap();
        }

        // to be consistent with the base topology
        ifc.mut_etags().for_each(|t| *t = 1);
        ifc.remove_faces(|t| self.is_partition_bdy(t));
        if self.debug {
            ifc.compute_face_to_elems();
            ifc.check_simple().unwrap();
        }

        let mut info = info.into_inner().unwrap();

        let topo = self.mesh.get_topology().unwrap().clone();
        ifc.compute_topology_from(topo);
        let ifc_m = ifc_m.into_inner().unwrap();

        let (mut ifc, ifc_m) = if let Some(dd_params) = dd_params.next(ifc.n_verts()) {
            let mesh = ifc;
            let mut dd = Self::new(mesh, ifc_m, self.n_parts)?;
            dd.set_debug(self.debug);
            dd.interface_bdy_tag = self.interface_bdy_tag + 1;
            let (ifc, interface_info, ifc_m) = dd.remesh(geom, params, &dd_params)?;
            info.interface = Some(Box::new(interface_info));
            (ifc, ifc_m)
        } else {
            debug!("Remeshing level {level} / interface");
            let mut ifc_remesher = Remesher::new(&ifc, &ifc_m, geom)?;
            if self.debug {
                ifc_remesher.check().unwrap();
            }
            let n_verts_init = ifc.n_verts();
            let now = Instant::now();
            ifc_remesher.remesh(&params, geom)?;
            info.interface = Some(Box::new(ParallelRemeshingInfo {
                info: RemeshingInfo {
                    n_verts_init,
                    n_verts_final: ifc_remesher.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                    ..Default::default()
                },
                partition_time: 0.0,
                partition_quality: 0.0,
                partition_imbalance: 0.0,
                partitions: Vec::new(),
                interface: None,
            }));
            (ifc_remesher.to_mesh(true), ifc_remesher.metrics())
        };

        if self.debug {
            let fname = format!("level_{level}_ifc_remeshed.vtu");
            ifc.vtu_writer().export(&fname).unwrap();
        }

        // Merge res and ifc
        let mut res = res.into_inner().unwrap();
        ifc.mut_etags().for_each(|t| *t = 2);
        let (ids, _, _) = res.add(&ifc, |_| true, |_| true, Some(1e-12));
        if self.debug {
            res.compute_face_to_elems();
            res.check_simple().unwrap();
        }
        let mut res_m = res_m.into_inner().unwrap();
        res_m.extend(ids.iter().map(|&i| ifc_m[i]));

        res.remove_faces(|t| self.is_interface_bdy(t));
        res.mut_etags().for_each(|t| *t = 1);
        if self.debug {
            res.compute_face_to_elems();
            res.check_simple().unwrap();
        }

        if self.debug {
            let fname = format!("level_{level}_final.vtu");
            res.vtu_writer().export(&fname).unwrap();
        }

        info.info.n_verts_final = res.n_verts();
        info.info.time = now.elapsed().as_secs_f64() + self.partition_time;

        Ok((res, info, res_m))
    }
}

#[cfg(test)]
mod tests {

    use crate::mesh::geom_elems::GElem;
    use tmesh::mesh::{
        Mesh,
        partition::{HilbertPartitioner, Partitioner},
    };

    use crate::{
        Idx, Result,
        geometry::NoGeometry,
        mesh::{HasTmeshImpl, Point, Triangle, test_meshes::test_mesh_2d},
        metric::IsoMetric,
        remesher::{
            ElementCostEstimator, ParallelRemesher, ParallelRemesherParams, RemesherParams,
            cost_estimator::{NoCostEstimator, TotoCostEstimator},
        },
    };

    fn test_domain_decomposition_2d<
        P: Partitioner,
        C: ElementCostEstimator<2, Triangle, IsoMetric<2>>,
    >(
        debug: bool,
        n_parts: Idx,
    ) -> Result<()> {
        // use crate::init_log;
        // init_log("debug");
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.mut_etags().for_each(|t| *t = 1);
        mesh.compute_topology();

        let h = |p: Point<2>| {
            let x = p[0];
            let y = p[1];
            let hmin = 0.001;
            let hmax = 0.1;
            let sigma: f64 = 0.25;
            hmin + (hmax - hmin)
                * (1.0 - f64::exp(-((x - 0.5).powi(2) + (y - 0.35).powi(2)) / sigma.powi(2)))
        };

        // let m: Vec<_> = (0..mesh.n_verts())
        //     .map(|i| IsoMetric::<2>::from(h(mesh.vert(i))))
        //     .collect();
        mesh.compute_volumes();
        let m: Vec<_> = mesh
            .gelems() // `gelems()` retourne un itérateur sur les éléments géométriques (GTriangle dans ce cas)
            .map(|g_elem| {
                let center_point = g_elem.center(); // Obtenez le Point<2> du centre de l'élément
                let metric_value = h(center_point); // Calculez la valeur scalaire (f64) de la métrique en utilisant ce point
                IsoMetric::<2>::from(metric_value) // Convertissez la valeur scalaire en IsoMetric<2>
            })
            .collect();
        println!("{}", m.len());
        let dd = ParallelRemesher::<2, Triangle, IsoMetric<2>, P, C>::new(mesh, m, n_parts)?;

        let dd_params = ParallelRemesherParams::new(2, 1, 0);
        let (mut mesh, _, _) = dd.remesh(&NoGeometry(), RemesherParams::default(), &dd_params)?;

        if debug {
            mesh.vtu_writer().export("res.vtu")?;
            mesh.vtu_writer().export("res_bdy.vtu")?;
        }

        let n = mesh.n_verts();
        for i in 0..n {
            let vi = mesh.vert(i);
            for j in i + 1..n {
                let vj = mesh.vert(j);
                let d = (vj - vi).norm();
                assert!(d > 1e-8, "{i}, {j}, {vi:?}, {vj:?}");
            }
        }

        mesh.compute_face_to_elems();
        mesh.check_simple()?;

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_dd_2d_hilbert_1() {
        test_domain_decomposition_2d::<
            HilbertPartitioner,
            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
        >(false, 1)
        .unwrap();
    }

    #[test]
    fn test_dd_2d_hilbert_2() -> Result<()> {
        test_domain_decomposition_2d::<
            HilbertPartitioner,
            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
        >(false, 2)
    }

    #[test]
    fn test_dd_2d_hilbert_3() -> Result<()> {
        test_domain_decomposition_2d::<
            HilbertPartitioner,
            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
        >(false, 3)
    }

    #[test]
    fn test_dd_2d_hilbert_4() -> Result<()> {
        test_domain_decomposition_2d::<
            HilbertPartitioner,
            TotoCostEstimator<2, Triangle, IsoMetric<2>>,
        >(true, 4)
    }

    #[test]
    fn test_dd_2d_hilbert_5() -> Result<()> {
        test_domain_decomposition_2d::<HilbertPartitioner, NoCostEstimator<2, Triangle, IsoMetric<2>>>(
            false, 5,
        )
    }

    #[cfg(feature = "metis")]
    #[test]
    #[should_panic]
    fn test_dd_2d_metis_1() {
        test_domain_decomposition_2d(false, PartitionType::MetisRecursive(1)).unwrap();
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_2() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::MetisRecursive(2))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_3() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::MetisRecursive(3))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_4() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::MetisRecursive(4))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_5() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::MetisRecursive(5))
    }

    // #[cfg(feature = "scotch")]
    // #[test]
    // #[should_panic]
    // fn test_dd_2d_scotch_1() {
    //     test_domain_decomposition_2d(false, PartitionType::Scotch(1)).unwrap();
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_2d_scotch_2() -> Result<()> {
    //     test_domain_decomposition_2d(false, PartitionType::Scotch(2))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_2d_scotch_3() -> Result<()> {
    //     test_domain_decomposition_2d(false, PartitionType::Scotch(3))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_2d_scotch_4() -> Result<()> {
    //     test_domain_decomposition_2d(false, PartitionType::Scotch(4))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_2d_scotch_5() -> Result<()> {
    //     test_domain_decomposition_2d(false, PartitionType::Scotch(5))
    // }

    // fn test_domain_decomposition_3d(debug: bool, ptype: PartitionType) -> Result<()> {
    //     // use crate::init_log;
    //     // init_log("warning");
    //     let mut mesh = test_mesh_3d().split().split().split();
    //     mesh.compute_topology();
    //     let dd = ParallelRemesher::new(mesh, ptype)?;
    //     // dd.set_debug(true);

    //     let h = |p: Point<3>| {
    //         let x = p[0];
    //         let y = p[1];
    //         let z = p[2];
    //         let hmin = 0.025;
    //         let hmax = 0.25;
    //         let sigma: f64 = 0.25;
    //         hmin + (hmax - hmin)
    //             * (1.0
    //                 - f64::exp(
    //                     -((x - 0.5).powi(2) + (y - 0.35).powi(2) + (z - 0.65).powi(2))
    //                         / sigma.powi(2),
    //                 ))
    //     };

    //     let m: Vec<_> = (0..dd.mesh.n_verts())
    //         .map(|i| IsoMetric::<3>::from(h(dd.mesh.vert(i))))
    //         .collect();

    //     let dd_params = ParallelRemesherParams::new(2, 2, 0);
    //     let (mut mesh, _, _) =
    //         dd.remesh(&m, &NoGeometry(), RemesherParams::default(), &dd_params)?;

    //     if debug {
    //         mesh.vtu_writer().export("res.vtu")?;
    //         mesh.vtu_writer().export("res_bdy.vtu")?;
    //     }

    //     let n = mesh.n_verts();
    //     for i in 0..n {
    //         let vi = mesh.vert(i);
    //         for j in i + 1..n {
    //             let vj = mesh.vert(j);
    //             let d = (vj - vi).norm();
    //             assert!(d > 1e-8, "{i}, {j}, {vi:?}, {vj:?}");
    //         }
    //     }
    //     mesh.compute_face_to_elems();
    //     mesh.check_simple()?;

    //     Ok(())
    // }

    // #[test]
    // #[should_panic]
    // fn test_dd_3d_hilbert_1() {
    //     test_domain_decomposition_3d(false, PartitionType::Hilbert(1)).unwrap();
    // }

    // #[test]
    // fn test_dd_3d_hilbert_2() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Hilbert(2))
    // }

    // #[test]
    // fn test_dd_3d_hilbert_3() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Hilbert(3))
    // }

    // #[test]
    // fn test_dd_3d_hilbert_4() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Hilbert(4))
    // }

    // #[test]
    // fn test_dd_3d_hilbert_5() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Hilbert(5))
    // }

    // #[cfg(feature = "metis")]
    // #[test]
    // #[should_panic]
    // fn test_dd_3d_metis_1() {
    //     test_domain_decomposition_3d(false, PartitionType::MetisRecursive(1)).unwrap();
    // }

    // #[cfg(feature = "metis")]
    // #[test]
    // fn test_dd_3d_metis_2() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::MetisRecursive(2))
    // }

    // #[cfg(feature = "metis")]
    // #[test]
    // fn test_dd_3d_metis_3() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::MetisRecursive(3))
    // }

    // #[cfg(feature = "metis")]
    // #[test]
    // fn test_dd_3d_metis_4() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::MetisRecursive(4))
    // }

    // #[cfg(feature = "metis")]
    // #[test]
    // fn test_dd_3d_metis_5() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::MetisRecursive(5))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // #[should_panic]
    // fn test_dd_3d_scotch_1() {
    //     test_domain_decomposition_3d(false, PartitionType::Scotch(1)).unwrap();
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_3d_scotch_2() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Scotch(2))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_3d_scotch_3() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Scotch(3))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_3d_scotch_4() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Scotch(4))
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_dd_3d_scotch_5() -> Result<()> {
    //     test_domain_decomposition_3d(false, PartitionType::Scotch(5))
    // }
}
