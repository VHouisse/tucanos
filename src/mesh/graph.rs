use crate::{
    mesh::{vector, Elem}, Error, Idx, Result, Tag
};
use num::PrimInt;
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, fmt::Display, ops::AddAssign};
use std::fmt;
/// Renumber the vertices in order to have contininuous indices, and return he map from old to nex indices
#[must_use]
pub fn reindex<E: Elem>(elems: &vector::Vector<E>) -> (Vec<E>, FxHashMap<Idx, Idx>) {
    let mut map = FxHashMap::default();
    let mut next = 0 as Idx;
    for i in elems.iter().flatten() {
        if let Entry::Vacant(e) = map.entry(i) {
            e.insert(next);
            next += 1;
        }
    }

    let new_elems = elems
        .iter()
        .map(|e| E::from_iter(e.iter().map(|&i| *map.get(&i).unwrap())))
        .collect();

    (new_elems, map)
}

#[derive(Debug, Default, Clone)]
pub struct CSRGraph {
    pub ptr: Vec<Idx>,
    pub indices: Vec<Idx>,
    pub m: Idx,
}

impl CSRGraph {
    fn from_edges(elems: &[[Idx; 2]]) -> Self {
        let nv = elems.iter().flatten().copied().max().unwrap_or(0) as usize + 1;

        let n = elems.iter().flatten().count();

        let mut res = Self {
            ptr: vec![0; nv + 1],
            indices: vec![Idx::MAX; n],
            m: 0,
        };

        for i in elems.iter().flatten().copied() {
            res.ptr[i as usize + 1] += 1;
        }

        for i in 0..nv {
            res.ptr[i + 1] += res.ptr[i];
        }

        res
    }

    fn from_elems<E: Elem>(elems: &vector::Vector<E>, nv: Option<usize>) -> Self {
        let nv = nv.unwrap_or_else(|| elems.iter().flatten().max().unwrap_or(0) as usize + 1);

        let n = elems.iter().flatten().count();

        let mut res = Self {
            ptr: vec![0; nv + 1],
            indices: vec![Idx::MAX; n],
            m: 0,
        };

        for i in elems.iter().flatten() {
            res.ptr[i as usize + 1] += 1;
        }

        for i in 0..nv {
            res.ptr[i + 1] += res.ptr[i];
        }

        res
    }

    pub fn sort(&mut self) {
        let n = self.ptr.len() - 1;
        for i in 0..n {
            let start = self.ptr[i] as usize;
            let end = self.ptr[i + 1] as usize;
            self.indices[start..end].sort_unstable();
        }
    }

    #[must_use]
    pub fn new(edgs: &[[Idx; 2]]) -> Self {
        let mut res = Self::from_edges(edgs);
        res.m = res.n();

        for e in edgs {
            let i0 = e[0] as usize;
            let i1 = e[1] as usize;
            let mut ok = false;
            for j in res.ptr[i0]..res.ptr[i0 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i1 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            let mut ok = false;
            for j in res.ptr[i1]..res.ptr[i1 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i0 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
        }
        res.sort();
        res
    }

    #[must_use]
    pub fn transpose<E: Elem>(elems: &vector::Vector<E>, nv: Option<usize>) -> Self {
        let mut res = Self::from_elems(elems, nv);
        res.m = elems.len() as Idx;

        for (i, e) in elems.iter().enumerate() {
            for i_vert in e.iter().copied() {
                let start = res.ptr[i_vert as usize];
                let end = res.ptr[i_vert as usize + 1];
                let mut ok = false;
                for j in start..end {
                    if res.indices[j as usize] == Idx::MAX {
                        res.indices[j as usize] = i as Idx;
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }
        res.sort();
        res
    }

    #[must_use]
    pub fn n(&self) -> Idx {
        self.ptr.len() as Idx - 1
    }

    #[must_use]
    pub const fn m(&self) -> Idx {
        self.m
    }

    #[must_use]
    pub fn n_edges(&self) -> Idx {
        self.indices.len() as Idx
    }

    #[must_use]
    pub fn row(&self, i: Idx) -> &[Idx] {
        let start = self.ptr[i as usize] as usize;
        let end = self.ptr[i as usize + 1] as usize;
        &self.indices[start..end]
    }
}

#[derive(Debug)]
pub struct ConnectedComponents<T: PrimInt + AddAssign + Display> {
    vtag: Vec<T>,
}

impl<T: PrimInt + AddAssign + Display> ConnectedComponents<T> {
    pub fn new(g: &CSRGraph) -> Result<Self> {
        assert_eq!(g.n(), g.m());

        let mut res = Self {
            vtag: vec![T::max_value(); g.n() as usize],
        };
        res.compute(g)?;
        Ok(res)
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn tags(&self) -> &[T] {
        &self.vtag
    }

    fn compute_from(&mut self, g: &CSRGraph, starts: &[Idx], component: T) {
        let mut next_starts = Vec::new();
        for &start in starts {
            self.vtag[start as usize] = component;
            for i in g.row(start).iter().copied() {
                if self.vtag[i as usize] == T::max_value() {
                    self.vtag[i as usize] = component;
                    next_starts.push(i);
                }
            }
        }
        if !starts.is_empty() {
            self.compute_from(g, &next_starts, component);
        }
    }

    fn compute(&mut self, g: &CSRGraph) -> Result<()> {
        let mut start = 0;
        let mut component = T::zero();
        while start < g.n() {
            self.compute_from(g, &[start], component);
            while start < g.n() && self.vtag[start as usize] < T::max_value() {
                start += 1;
            }
            component += T::one();
            if component == T::max_value() {
                return Err(Error::from("too many connected components"));
            }
            assert!(component < T::max_value());
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct ConnectedComponentsInfo{
    pub cc_idx : Idx,
    pub elements : Vec<Idx>,
    pub total_work : f64,
    pub is_primary : bool,
    pub partition_id : Tag,
}

impl ConnectedComponentsInfo{
    #[must_use]
    pub fn new(
        cc_idx : Idx,
        elements : Vec<Idx>,
        total_work : f64,
        is_primary: bool,
        partition_id : Tag,
    )->Self{ 
        Self{
        cc_idx,
        elements,
        total_work,
        is_primary,
        partition_id,
        }
    }
}
impl fmt::Display for ConnectedComponentsInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Vous pouvez choisir le format qui vous convient le mieux.
        // Voici un exemple qui affiche tous les champs de manière lisible.
        write!(
            f,
            " ConnectedComponentInfo {{ cc_idx: {}, elements: {}, total_work: {:.2}, is_primary: {}, partition_id: {} }} ",
            self.cc_idx,
            self.elements.len(),
            self.total_work,
            self.is_primary,
            self.partition_id
        )
    }
}

impl Clone for ConnectedComponentsInfo {
    fn clone(&self) -> Self {
        Self {
            cc_idx: self.cc_idx,
            elements: self.elements.clone(), // Clone le Vec<Idx>
            total_work: self.total_work,
            is_primary: self.is_primary,
            partition_id: self.partition_id,
        }
    }
}

impl ConnectedComponentsInfo{
    pub fn get_cc_idx(&self) -> Idx{
        self.cc_idx 
    }

    pub fn get_is_primary(&self)->bool{
        self.is_primary
    }
    pub fn elements_iter(&self) -> impl Iterator<Item = Idx> + '_ {
        self.elements.iter().copied()
    }
    pub fn get_partition_id(&self)->Tag{
        self.partition_id
    }
    pub fn get_total_work(&self)->f64{
        self.total_work
    }
}


#[cfg(test)]
mod tests {
    use crate::{
        Tag,
        mesh::{
            Edge, Elem, Triangle,
            graph::{CSRGraph, ConnectedComponents, reindex},
        },
    };

    #[test]
    fn test_reindex() {
        let g = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 0]),
            Edge::from_slice(&[5, 6]),
        ];
        let (new_elems, _) = reindex(&g.into());
        assert_eq!(new_elems.iter().copied().flatten().max().unwrap(), 4);
        assert_eq!(new_elems.len(), 4);
    }

    #[test]
    fn test_csr_edges() {
        let g = vec![[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::new(&g);
        assert_eq!(g.n(), 5);
        assert_eq!(g.m(), 5);
        assert_eq!(g.n_edges(), 8);
        let edgs = g.row(0);
        assert_eq!(*edgs, [1, 2]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0, 2]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [4]);
        let edgs = g.row(4);
        assert_eq!(*edgs, [3]);
    }

    #[test]
    fn test_csr_triangles() {
        let g = vec![
            Triangle::from_slice(&[0, 1, 2]),
            Triangle::from_slice(&[0, 2, 3]),
        ];

        let g = CSRGraph::transpose(&g.into(), None);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 2);

        let edgs = g.row(0);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [1]);
    }

    #[test]
    fn test_cc() {
        let g = vec![[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::new(&g);
        let cc = ConnectedComponents::<Tag>::new(&g).unwrap();
        assert_eq!(cc.tags(), [0, 0, 0, 1, 1]);
    }
}
