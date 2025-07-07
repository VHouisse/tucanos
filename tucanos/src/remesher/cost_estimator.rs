use rayon::iter::ParallelIterator;
use std::marker::PhantomData;

use crate::{
    mesh::{Elem, SimplexMesh},
    metric::{AnisoMetric2d, HasImpliedMetric, IsoMetric, Metric},
};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;

pub trait ElementCostEstimator<const D: usize, E: Elem, M: Metric<D>>: Send + Sync {
    // fn new(msh: &SimplexMesh<D, E>, m: &[M]) -> Self;
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
    // fn new(msh: &SimplexMesh<D, E>, _m: &[M]) -> Self {
    //     Self {
    //         n_elems: msh.n_elems(),
    //         _e: PhantomData,
    //         _m: PhantomData,
    //     }
    // }

    fn compute(&self, msh: &SimplexMesh<D, E>, _m: &[M]) -> Vec<f64> {
        vec![1.0; msh.n_elems() as usize]
    }
    type CurrentImpliedMetricType = AnisoMetric2d;
}

pub struct TotoCostEstimator<const D: usize, E: Elem, M: Metric<D>>
where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
    M: Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
{
    _e: PhantomData<E>,
    _m: PhantomData<M>,
}

fn work_eval(initial_density: f64, actual_density: f64, intersected_density: f64, vol: f64) -> f64 {
    // Set up to csts and evaluate real coeff
    let insert_c: f64 = 1.0;
    let collapse_c: f64 = 1.3;
    let optimization_c: f64 = 3.3;
    let work = vol
        * (insert_c * (intersected_density - initial_density)
            + collapse_c * (intersected_density - actual_density)
            + optimization_c * actual_density);
    work
}

impl<const D: usize, E: Elem, M: Metric<D>> TotoCostEstimator<D, E, M>
where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
    M: Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
{
    pub fn new() -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> ElementCostEstimator<D, E, M>
    for TotoCostEstimator<D, E, M>
where
    E::Geom<D, IsoMetric<D>>: HasImpliedMetric<D, IsoMetric<D>>,
    M: Into<<E::Geom<D, IsoMetric<D>> as HasImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
{
    // fn new(msh: &SimplexMesh<D, E>, _m: &[M]) -> Self {
    //     Self {
    //         n_elems: msh.n_elems(),
    //         _e: PhantomData,
    //         _m: PhantomData,
    //     }
    // }
    type CurrentImpliedMetricType = <<E as Elem>::Geom<D, IsoMetric<D>> as HasImpliedMetric<
        D,
        IsoMetric<D>,
    >>::ImpliedMetricType;

    fn compute(&self, msh: &SimplexMesh<D, E>, m: &[M]) -> Vec<f64> {
        let weights: Vec<_> = msh
            .par_gelems() // Commence avec l'itérateur parallèle des éléments géométriques
            .zip(m.par_iter()) // Zipe avec l'itérateur parallèle des métriques initiales
            .zip(msh.get_elem_volumes().unwrap().par_iter()) // Zipe avec les volumes (en parallèle aussi pour cohérence)
            // Le 'map' suivant traite un tuple: ((ge, p_m_ref), vol_ref)
            .map(|((ge, p_m_ref), vol_ref)| {
                let implied_metric = ge.calculate_implied_metric(); // Calcul de la métrique implicite
                let converted_p_m: Self::CurrentImpliedMetricType = (*p_m_ref).into(); // Conversion de la métrique initiale
                let intersected_metric = implied_metric.intersect(&converted_p_m); // Intersection des métriques

                // Calcul des densités
                let d_initial_metric = p_m_ref.density();
                let d_actual_metric = implied_metric.density();
                let d_intersected = intersected_metric.density();

                work_eval(d_initial_metric, d_actual_metric, d_intersected, *vol_ref) // Appel de work_eval
            })
            .collect(); // Collecte le résultat final dans un Vec

        weights
    }
}
