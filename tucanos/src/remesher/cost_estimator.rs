use rayon::iter::ParallelIterator;
use std::marker::PhantomData;

use crate::{
    mesh::{Elem, SimplexMesh},
    metric::{AnisoMetric2d, HasImpliedMetric, IsoMetric, Metric},
};

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
        let implied_metrics: Vec<Self::CurrentImpliedMetricType> = msh
            .par_gelems()
            .map(|ge| ge.calculate_implied_metric())
            .collect();

        assert_eq!(implied_metrics.len(), m.len());
        let intersected_metrics: Vec<Self::CurrentImpliedMetricType> = implied_metrics
            .iter()
            .zip(m.iter())
            .map(|(implied_metrics_ref, _p_m_ref)| {
                let tmp = *_p_m_ref;
                let tmp: Self::CurrentImpliedMetricType = tmp.into();
                implied_metrics_ref.intersect(&tmp)
            })
            .collect();

        let d_initial_metric: Vec<_> = m
            .iter()
            .map(|_p_metric_ref| _p_metric_ref.density())
            .collect();

        let d_actual_metric: Vec<_> = implied_metrics
            .iter()
            .map(|implied_metrics_ref| implied_metrics_ref.density())
            .collect();

        let d_intersected: Vec<_> = intersected_metrics
            .iter()
            .map(|_intersected_metric_ref| _intersected_metric_ref.density())
            .collect();
        let volumes = msh.get_elem_volumes().unwrap().to_vec();

        let weights: Vec<_> = d_initial_metric
            .iter()
            .zip(d_actual_metric.iter())
            .zip(d_intersected.iter())
            .zip(volumes.iter())
            .map(|(((_d_i_m, _d_a_m), _d_id), vol)| work_eval(*_d_i_m, *_d_a_m, *_d_id, *vol))
            .collect();

        weights
    }
}
