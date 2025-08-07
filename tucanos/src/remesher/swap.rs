use std::time::Instant;

use super::Remesher;
use crate::{
    Dim, Idx, Result,
    geometry::Geometry,
    mesh::Elem,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{StepStats, SwapStats},
    },
};
use log::{debug, trace};

pub enum TrySwapResult {
    CouldNotSwap,
    CouldSwap,
    QualitySufficient,
    FixedEdge,
}

#[derive(Clone, Debug)]
pub struct SwapParams {
    /// Quality below which swap is applied
    pub q: f64,
    /// Max. number of loops through the mesh edges during the swap step
    pub max_iter: u32,
    /// Constraint the length of the newly created edges to be < swap_max_l_rel * max(l) during swap
    pub max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < swap_max_l_abs during swap
    pub max_l_abs: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_rel * min(l) during swap
    pub min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_abs during swap
    pub min_l_abs: f64,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
}

impl Default for SwapParams {
    fn default() -> Self {
        Self {
            q: 0.8,
            max_iter: 2,
            max_l_rel: 1.5,
            max_l_abs: 1.5 * f64::sqrt(2.0),
            min_l_rel: 0.75,
            min_l_abs: 0.75 / f64::sqrt(2.0),
            max_angle: 25.0,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> Remesher<D, E, M> {
    /// Try to swap an edge if
    ///   - one of the elements in its cavity has a quality < qmin
    ///   - no edge smaller that `l_min` or longer that `l_max` is created
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn try_swap<G: Geometry<D>>(
        &mut self,
        edg: [Idx; 2],
        params: &SwapParams,
        max_angle: f64,
        cavity: &mut Cavity<D, E, M>,
        geom: &G,
    ) -> Result<(TrySwapResult, f64)> {
        let start_time = Instant::now();
        trace!("Try to swap edge {edg:?}");
        cavity.init_from_edge(edg, self);
        if cavity.global_elem_ids.len() == 1 {
            trace!("Cannot swap, only one adjacent cell");
            return Ok((
                TrySwapResult::QualitySufficient,
                start_time.elapsed().as_secs_f64(),
            ));
        }

        if cavity.q_min > params.q {
            trace!("No need to swap, quality sufficient");
            return Ok((
                TrySwapResult::QualitySufficient,
                start_time.elapsed().as_secs_f64(),
            ));
        }

        let l_min = params.min_l_abs.min(params.min_l_rel * cavity.l_min);
        let l_max = params.max_l_abs.min(params.max_l_rel * cavity.l_max);

        trace!("min. / max. allowed edge length = {l_min:.2}, {l_max:.2}");

        let Seed::Edge(local_edg) = cavity.seed else {
            unreachable!()
        };
        let local_i0 = local_edg[0] as usize;
        let local_i1 = local_edg[1] as usize;
        let mut q_ref = cavity.q_min;

        let mut vx = 0;
        let mut succeed = false;

        let etag = self
            .topo
            .parent(cavity.tags[local_i0], cavity.tags[local_i1])
            .unwrap();
        // tag < 0 on fixed boundaries
        if etag.1 < 0 {
            return Ok((TrySwapResult::FixedEdge, start_time.elapsed().as_secs_f64()));
        }

        if etag.0 < E::Face::DIM as Dim {
            return Ok((
                TrySwapResult::CouldNotSwap,
                start_time.elapsed().as_secs_f64(),
            ));
        }

        for n in 0..cavity.n_verts() {
            if n == local_i0 as Idx || n == local_i1 as Idx {
                continue;
            }

            // check topo
            let ptag = self.topo.parent(etag, cavity.tags[n as usize]);
            if ptag.is_none() {
                trace!("Cannot swap, incompatible geometry");
                continue;
            }
            let ptag = ptag.unwrap();
            if ptag.0 != etag.0 || ptag.1 != etag.1 {
                trace!("Cannot swap, incompatible geometry");
                continue;
            }

            // too difficult otherwise!
            if !cavity.tagged_faces.is_empty() {
                assert!(cavity.tagged_faces.len() == 2);
                if !cavity.tagged_faces().any(|(f, _)| f.contains_vertex(n)) {
                    continue;
                }
            }

            let ftype = FilledCavityType::ExistingVertex(n);
            let filled_cavity = FilledCavity::new(cavity, ftype);

            if filled_cavity.is_same() {
                continue;
            }

            if !filled_cavity.check_boundary_normals(&self.topo, geom, max_angle) {
                trace!("Cannot swap, would create a non smooth surface");
                continue;
            }

            if let CavityCheckStatus::Ok(min_quality) = filled_cavity.check(l_min, l_max, q_ref) {
                trace!("Can swap  from {n} : ({min_quality} > {q_ref})");
                succeed = true;
                q_ref = min_quality;
                vx = n;
            }
        }

        if succeed {
            trace!("Swap from {vx}");
            let ftype = FilledCavityType::ExistingVertex(vx);
            let filled_cavity = FilledCavity::new(cavity, ftype);
            for e in &cavity.global_elem_ids {
                self.remove_elem(*e)?; // L'opérateur ? fonctionne à nouveau
            }
            let global_vx = cavity.local2global[vx as usize];
            for (f, t) in filled_cavity.faces() {
                let f = cavity.global_elem(&f);
                assert!(!f.contains_vertex(global_vx));
                assert!(!f.contains_edge(edg));
                let e = E::from_vertex_and_face(global_vx, &f);
                self.insert_elem(e, t)?; // L'opérateur ? fonctionne à nouveau
            }
            for (f, _) in cavity.global_tagged_faces() {
                self.remove_tagged_face(f)?; // L'opérateur ? fonctionne à nouveau
            }
            for (b, t) in filled_cavity.tagged_faces_boundary_global() {
                self.add_tagged_face(E::Face::from_vertex_and_face(global_vx, &b), t)?; // L'opérateur ? fonctionne à nouveau
            }

            return Ok((TrySwapResult::CouldSwap, start_time.elapsed().as_secs_f64()));
        }

        Ok((
            TrySwapResult::CouldNotSwap,
            start_time.elapsed().as_secs_f64(),
        ))
    }

    pub fn swap<G: Geometry<D>>(
        &mut self,
        params: &SwapParams,
        geom: &G,
        debug: bool,
    ) -> Result<u32> {
        debug!("Swap edges: target quality = {}", params.q);
        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;
            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut n_swaps = 0;
            let mut n_fails = 0;
            let mut n_verifs_attempted = 0;

            let mut total_success_time = 0.0;
            let mut total_fail_time = 0.0;
            let mut total_verif_time = 0.0;

            for edg in edges {
                // Ici, nous utilisons l'opérateur ? sur le Result retourné par try_swap
                // et nous déstructurons le tuple (TrySwapResult, f64)
                let (res, time_spent) =
                    self.try_swap(edg, params, params.max_angle, &mut cavity, geom)?;
                match res {
                    TrySwapResult::CouldNotSwap => {
                        n_fails += 1;
                        total_fail_time += time_spent;
                    }
                    TrySwapResult::CouldSwap => {
                        n_swaps += 1;
                        total_success_time += time_spent;
                    }
                    TrySwapResult::QualitySufficient | TrySwapResult::FixedEdge => {
                        n_verifs_attempted += 1;
                        total_verif_time += time_spent;
                    }
                }
            }

            debug!(
                "Iteration {n_iter}: {n_swaps} edges swapped ({n_fails} failed, {n_verifs_attempted} checked/ok)"
            );

            if n_swaps == 0 || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                self.stats.push(StepStats::Swap(SwapStats::new(
                    n_swaps,
                    n_fails,
                    n_verifs_attempted,
                    total_success_time,
                    total_fail_time,
                    total_verif_time,
                    self,
                )));
                return Ok(n_iter);
            }
        }
    }
}
