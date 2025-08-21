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
    let insert_c: f64 = 8.03; // Cout du split impliquant la création de 6 arêtes moyennes => 6 swaps verifs 
    let collapse_c: f64 = 3.33; // Cout du collapse impliquant la suppresion de 6 arêtes moy => 6 swaps verifs 
    let verif_cost = 1.0;
    let insert_prob = intersected_density / initial_density;
    let collapse_prob = intersected_density / actual_density;
    println!("Insert density : {insert_prob}");
    println!("Collapse density : {collapse_prob}");
    vol * (insert_c * (intersected_density - initial_density)
        + collapse_c * (intersected_density - actual_density))
        + verif_cost * 4.0 // Multiplicateur nb arêtes per elems 
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
        let weights: Vec<f64> = msh
            .par_elems()
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
            .collect();
        let (min, max, _sum) = weights.iter().fold((f64::MAX, f64::MIN, 0.0), |a, &b| {
            (a.0.min(b), a.1.max(b), a.2 + b)
        });
        let diff = max - min;
        let fixed_weight = diff * (1.0 / f64::from(msh.n_verts())).cbrt();
        println!("Fixed Weight {fixed_weight}");
        weights.iter().map(|&w| w + fixed_weight).collect()

        // let sum_work: f64 = work.iter().sum();
    }
}
