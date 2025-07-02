use crate::{
    mesh::{Elem, SimplexMesh},
    metric::Metric,
};

pub trait ElementCostEstimator<const D: usize, E: Elem, M: Metric<D>>: Send + Sync {
    fn new(msh: &SimplexMesh<D, E>, m: &[M]) -> Self;
    fn compute(&self) -> Vec<f64>;
}
