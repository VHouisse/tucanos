use std::marker::PhantomData;

use crate::{
    Idx,
    mesh::{Elem, SimplexMesh},
    metric::Metric,
};

pub trait ElementCostEstimator<const D: usize, E: Elem, M: Metric<D>>: Send + Sync {
    fn new(msh: &SimplexMesh<D, E>, m: &[M]) -> Self;
    fn compute(&self) -> Vec<f64>;
}

pub struct NoCostEstimator<const D: usize, E: Elem, M: Metric<D>> {
    n_elems: Idx,
    _e: PhantomData<E>,
    _m: PhantomData<M>,
}

impl<const D: usize, E: Elem, M: Metric<D>> ElementCostEstimator<D, E, M>
    for NoCostEstimator<D, E, M>
{
    fn new(msh: &SimplexMesh<D, E>, _m: &[M]) -> Self {
        Self {
            n_elems: msh.n_elems(),
            _e: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self) -> Vec<f64> {
        vec![1.0; self.n_elems as usize]
    }
}
