use core::f64;
use rayon::iter::ParallelIterator;
use std::marker::PhantomData;

use crate::mesh::geom_elems::GElem;
use crate::{
    mesh::{Elem, SimplexMesh},
    metric::{AnisoMetric2d, HasImpliedMetric, IsoMetric, Metric},
};
// const ADD_PERCENTAGE: f64 = 40.0;
pub trait ElementCostEstimator<const D: usize, E: Elem, M: Metric<D>>: Send + Sync {
    fn new(m: &[M]) -> Self;
    // fn compute(&self) -> Vec<f64>;
    fn compute(&self, msh: &SimplexMesh<D, E>, m: &[M]) -> Vec<f64>;
    type CurrentImpliedMetricType;
}

pub struct NoCostEstimator<const D: usize, E: Elem, M: Metric<D>> {
    _e: PhantomData<E>,
    _m: PhantomData<M>,
}

impl<const D: usize, E: Elem, M: Metric<D>> ElementCostEstimator<D, E, M>
    for NoCostEstimator<D, E, M>
{
    fn new(_m: &[M]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &SimplexMesh<D, E>, _m: &[M]) -> Vec<f64> {
        vec![1.0; msh.n_elems() as usize]
    }
    type CurrentImpliedMetricType = AnisoMetric2d;
}

pub struct TotoCostEstimator<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
> where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    _e: PhantomData<E>,
    _m: PhantomData<M>,
}

fn work_eval(initial_density: f64, actual_density: f64, intersected_density: f64, vol: f64) -> f64 {
    // Set up to csts and evaluate real coeff
    let insert_c: f64 = 0.365; // Un split dure en moyenne 3.65 * plus longtemps qu'une vérif
    let collapse_c: f64 = 0.8395; // Un collpase dure en moyenne 2.3 * plus longtemps qu'une vérif 2.3*3.65 = 8.395 
    let verif_cost = 0.1; // Cout unitaire d'une vérif (basé sur le temps de vérif d'un swap)
    let insert_prop = intersected_density - initial_density;
    let insert_bool = if insert_prop < 0.1 { 1.0 } else { 0.0 };

    let collapse_prop = intersected_density - actual_density;
    let collapse_bool = if collapse_prop < 0.1 { 1.0 } else { 0.0 };

    vol * ((insert_c + 0.10 * collapse_c + 6.0 * verif_cost) * insert_prop // Un split provoque 0.10 collapse et 6 nouvelles arrêtes
        + (collapse_c + 0.78 * insert_c - 6.0 * verif_cost) * collapse_prop)// Un collapse provoque 0.78 split et supprime 6 arrêtes 
        + verif_cost * (2.0 + collapse_bool + insert_bool) // Si collapse ou split, on ne fait pas la vérification correspondante
}

#[allow(clippy::new_without_default)]
impl<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
> TotoCostEstimator<D, E, M>
where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    #[must_use]
    pub const fn new(_m: &[M]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }
}

impl<
    const D: usize,
    E: Elem,
    M: Metric<D>
        + Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
> ElementCostEstimator<D, E, M> for TotoCostEstimator<D, E, M>
where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
{
    fn new(_m: &[M]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }
    type CurrentImpliedMetricType = <<E as Elem>::Geom<D, IsoMetric<D>> as HasImpliedMetric<
        D,
        IsoMetric<D>,
    >>::ImpliedMetricType;

    fn compute(&self, msh: &SimplexMesh<D, E>, m: &[M]) -> Vec<f64> {
        msh.par_elems()
            .map(|e| {
                let ge = msh.gelem(e);
                let implied_metric = ge.calculate_implied_metric();
                let vol = ge.vol();
                let weight = 1.0 / f64::from(E::N_VERTS);
                let mean_target_metric =
                    M::interpolate(e.iter().map(|i| (weight, &m[*i as usize])));
                let mean_target_metric: Self::CurrentImpliedMetricType = mean_target_metric.into();
                let intersected_metric = implied_metric.intersect(&mean_target_metric);
                let d_initial_metric = implied_metric.density();
                let d_target_metric = mean_target_metric.density();
                let d_intersected = intersected_metric.density();
                work_eval(d_initial_metric, d_target_metric, d_intersected, vol)
            })
            .collect()
        // let total_work_sum: f64 = weights.iter().sum();
        // let work_center_x = msh
        //     .gelems()
        //     .zip(weights.iter())
        //     .map(|(elem, &w)| elem.center()[0] * w)
        //     .sum::<f64>()
        //     / total_work_sum;

        // let work_center_y = msh
        //     .gelems()
        //     .zip(weights.iter())
        //     .map(|(elem, &w)| elem.center()[1] * w)
        //     .sum::<f64>()
        //     / total_work_sum;

        // let work_center_z = msh
        //     .gelems()
        //     .zip(weights.iter())
        //     .map(|(elem, &w)| elem.center()[2] * w)
        //     .sum::<f64>()
        //     / total_work_sum;

        // let work_center = Vector3::new(work_center_x, work_center_y, work_center_z);
        // println!("wok_center{work_center:?}");

        // let work_inertia_moment = msh
        //     .gelems()
        //     .zip(weights.iter())
        //     .map(|(elem, &w)| {
        //         let dx = elem.center()[0] - work_center[0];
        //         let dy = elem.center()[1] - work_center[1];
        //         let dz = elem.center()[2] - work_center[2];
        //         let distance_sq = dx.powi(2) + dy.powi(2) + dz.powi(2);
        //         w * distance_sq
        //     })
        //     .sum::<f64>()
        //     / total_work_sum;

        // let concentration_factor = work_inertia_moment.sqrt();
        // // Calcul du facteur de concentration

        // println!("Inertia Moment of the distribution : : : : {concentration_factor}");

        // let (min, max, _sum) = weights.iter().fold((f64::MAX, f64::MIN, 0.0), |a, &b| {
        //     (a.0.min(b), a.1.max(b), a.2 + b)
        // });
        // let diff = max - min;
        // let fixed_weight = diff * (1.0 / f64::from(msh.n_verts())).cbrt();
        // println!("Fixed Weight {fixed_weight}");
        // weights.iter().map(|&w| w + fixed_weight).collect()

        // let sum_work: f64 = work.iter().sum();
    }

    // fn locality_correction(weights: Vec<f64>, mesh: &SimplexMesh<D, E>) {}
}
