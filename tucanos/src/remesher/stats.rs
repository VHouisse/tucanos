use crate::{Idx, mesh::Elem, metric::Metric, remesher::Remesher};
use core::fmt;
use serde::Serialize;

/// Simple statistics (histogram + mean) to be used on edge lengths and element qualities
#[derive(Serialize, Clone, Debug)]
pub struct Stats {
    /// Histogram bins (length = n+1)
    pub bins: Vec<f64>,
    /// Histogram values (length = n)
    pub vals: Vec<f64>,
    pub mean: f64,
}

impl Stats {
    /// Compute the stats
    /// the bins use the minimum / maximum of the iterator as first and last values, and values in between
    pub fn new<I: Iterator<Item = f64>>(f: I, values: &[f64]) -> Self {
        let mut mini = f64::INFINITY;
        let mut maxi = f64::NEG_INFINITY;
        let mut count = 0;

        let n = values.len();

        let mut bins = vec![0.0; n + 2];
        bins[0] = mini;
        bins[1..=n].copy_from_slice(&values[..(n + 1 - 1)]);
        bins[n + 1] = maxi;
        let mut vals = vec![0.0; n + 1];
        let mut mean = 0.;
        for val in f {
            mini = mini.min(val);
            maxi = maxi.max(val);
            count += 1;
            mean += val;
            if val < bins[1] {
                vals[0] += 1.0;
            } else if val > bins[n] {
                vals[n] += 1.;
            } else {
                for i in 1..n {
                    if val > bins[i] && val <= bins[i + 1] {
                        vals[i] += 1.0;
                        break;
                    }
                }
            }
        }

        loop {
            let mut stop = true;
            for (i, val) in vals.iter().enumerate() {
                if *val < 0.5 {
                    bins.remove(i + 1);
                    vals.remove(i);
                    stop = false;
                    break;
                }
            }
            if stop {
                break;
            }
        }

        for val in &mut vals {
            *val /= f64::from(count);
        }

        let n = bins.len();
        bins[0] = mini;
        bins[n - 1] = maxi;
        mean /= f64::from(count);

        Self { bins, vals, mean }
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mean = {:.2}", self.mean)?;
        let n = self.bins.len() - 1;
        for i in 0..n {
            write!(
                f,
                ", {:.2} < {:.1}% < {:.2}",
                self.bins[i],
                100.0 * self.vals[i],
                self.bins[i + 1]
            )?;
        }
        Ok(())
    }
}

/// Statistics on the remesher state
#[derive(Serialize, Clone, Debug)]
pub struct RemesherStats {
    /// The # of vertices in the mesh
    n_verts: Idx,
    /// The # of elements in the mesh
    n_elems: Idx,
    /// The # of edges in the mesh
    n_edges: Idx,
    /// Edge length stats
    stats_l: Stats,
    /// Element quality stats
    stats_q: Stats,
}

impl RemesherStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(r: &Remesher<D, E, M>) -> Self {
        Self {
            n_verts: r.n_verts(),
            n_elems: r.n_elems(),
            n_edges: r.n_edges(),
            stats_l: Stats::new(r.lengths_iter(), &[f64::sqrt(0.5), f64::sqrt(2.0)]),
            stats_q: Stats::new(r.qualities_iter(), &[0.4, 0.6, 0.8]),
        }
    }
}

/// Statistics for each remeshing step that include `RemesherStats` and additional step-dependent info
#[derive(Serialize, Clone, Debug)]
pub enum StepStats {
    Init(InitStats),
    Split(SplitStats),
    Swap(SwapStats),
    Collapse(CollapseStats),
    Smooth(SmoothStats),
}
impl fmt::Display for StepStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Init(_s) => write!(f, "None"),
            Self::Split(s) => write!(f, "{s}"),
            Self::Swap(s) => write!(f, "{s}"),
            Self::Collapse(s) => write!(f, "{s}"),
            Self::Smooth(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct InitStats {
    r_stats: RemesherStats,
}

impl InitStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(r: &Remesher<D, E, M>) -> Self {
        Self {
            r_stats: RemesherStats::new(r),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SplitStats {
    n_splits: Idx,
    n_fails: Idx,
    n_verifs: Idx,
    r_stats: RemesherStats,
    total_verif_time: f64,
    total_split_time: f64,
    total_fails_time: f64,
}

impl SplitStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(
        n_splits: Idx,
        n_fails: Idx,
        n_verifs: Idx,
        r: &Remesher<D, E, M>,
        total_verif_time: f64,
        total_split_time: f64,
        total_fails_time: f64,
    ) -> Self {
        Self {
            n_splits,
            n_fails,
            n_verifs,
            r_stats: RemesherStats::new(r),
            total_verif_time,
            total_split_time,
            total_fails_time,
        }
    }
    pub const fn get_n_splits(&self) -> Idx {
        self.n_splits
    }

    pub const fn get_n_fails(&self) -> Idx {
        self.n_fails
    }
    pub const fn get_n_verifs(&self) -> Idx {
        self.n_verifs
    }

    pub const fn get_t_time_split(&self) -> f64 {
        self.total_split_time
    }

    pub const fn get_t_time_fails(&self) -> f64 {
        self.total_fails_time
    }

    pub const fn get_t_time_verif(&self) -> f64 {
        self.total_verif_time
    }

    pub const fn get_avg_time_split(&self) -> f64 {
        if self.n_splits == 0 {
            0.0
        } else {
            self.total_split_time / self.n_splits as f64
        }
    }

    pub const fn get_avg_time_fails(&self) -> f64 {
        if self.n_fails == 0 {
            0.0
        } else {
            self.total_fails_time / self.n_fails as f64
        }
    }

    pub const fn get_avg_time_verif(&self) -> f64 {
        if self.n_verifs == 0 {
            0.0
        } else {
            self.total_verif_time / self.n_verifs as f64
        }
    }
}
impl fmt::Display for SplitStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //writeln!(f, "  Split Stats:")?;
        writeln!(f, "    Splits Succeeded: {}", self.n_splits)?;
        writeln!(f, "    Splits Failed:    {}", self.n_fails)?;
        writeln!(f, "    Verifs:    {}", self.n_verifs)?;

        Ok(())
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SwapStats {
    n_swaps: Idx,
    n_fails: Idx,
    n_verifs: Idx,
    total_success_time: f64,
    total_fail_time: f64,
    total_verif_time: f64,
    r_stats: RemesherStats,
}

impl SwapStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(
        n_swaps: Idx,
        n_fails: Idx,
        n_verifs: Idx,
        total_success_time: f64,
        total_fail_time: f64,
        total_verif_time: f64,
        r: &Remesher<D, E, M>,
    ) -> Self {
        Self {
            n_swaps,
            n_fails,
            n_verifs,
            total_success_time,
            total_fail_time,
            total_verif_time,
            r_stats: RemesherStats::new(r),
        }
    }

    pub const fn get_n_swaps(&self) -> Idx {
        self.n_swaps
    }

    pub const fn get_n_fails(&self) -> Idx {
        self.n_fails
    }

    pub const fn get_n_verifs(&self) -> Idx {
        self.n_verifs
    }

    pub const fn get_t_time_swaps_success(&self) -> f64 {
        self.total_success_time
    }

    pub const fn get_avg_time_swaps_success(&self) -> f64 {
        if self.n_swaps - self.n_fails == 0 {
            0.0
        } else {
            self.total_success_time / (self.n_swaps - self.n_fails) as f64
        }
    }

    pub const fn get_t_time_swaps_fails(&self) -> f64 {
        self.total_fail_time
    }

    pub const fn get_avg_time_swaps_fails(&self) -> f64 {
        if self.n_fails == 0 {
            0.0
        } else {
            self.total_fail_time / self.n_fails as f64
        }
    }

    pub const fn get_t_time_swaps_verif(&self) -> f64 {
        self.total_verif_time
    }

    pub const fn get_avg_time_swaps_verif(&self) -> f64 {
        if self.n_verifs == 0 {
            0.0
        } else {
            self.total_verif_time / self.n_verifs as f64
        }
    }

    pub const fn get_total_exec_time(&self) -> f64 {
        self.total_success_time + self.total_fail_time + self.total_verif_time
    }
}
impl fmt::Display for SwapStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //writeln!(f, "  Swap Stats:")?;
        writeln!(f, "    Swaps Succeeded:  {}", self.n_swaps)?;
        writeln!(f, "    Swaps Failed:     {}", self.n_fails)?;
        Ok(())
    }
}
#[derive(Serialize, Clone, Debug)]
pub struct CollapseStats {
    n_collapses: Idx,
    n_fails: Idx,
    n_verifs: Idx,
    r_stats: RemesherStats,
    total_verif_time: f64,
    total_collapse_time: f64,
    total_fails_time: f64,
}

impl CollapseStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(
        n_collapses: Idx,
        n_fails: Idx,
        n_verifs: Idx,
        r: &Remesher<D, E, M>,
        total_verif_time: f64,
        total_collapse_time: f64,
        total_fails_time: f64,
    ) -> Self {
        Self {
            n_collapses,
            n_fails,
            n_verifs,
            r_stats: RemesherStats::new(r),
            total_verif_time,
            total_collapse_time,
            total_fails_time,
        }
    }

    pub const fn get_n_collapses(&self) -> Idx {
        self.n_collapses
    }

    pub const fn get_n_fails(&self) -> Idx {
        self.n_fails
    }

    pub const fn get_n_verifs(&self) -> Idx {
        self.n_verifs
    }

    pub const fn get_avg_time_collapse(&self) -> f64 {
        if self.n_collapses == 0 {
            0.0
        } else {
            self.total_collapse_time / self.n_collapses as f64
        }
    }

    pub const fn get_t_time_collapse(&self) -> f64 {
        self.total_collapse_time
    }

    pub const fn get_avg_time_fails(&self) -> f64 {
        if self.n_fails == 0 {
            0.0
        } else {
            self.total_fails_time / self.n_fails as f64
        }
    }

    pub const fn get_t_time_fails(&self) -> f64 {
        self.total_fails_time
    }

    pub const fn get_avg_time_verif(&self) -> f64 {
        if self.n_verifs == 0 {
            0.0
        } else {
            self.total_verif_time / self.n_verifs as f64
        }
    }

    pub const fn get_t_time_verif(&self) -> f64 {
        self.total_verif_time
    }
}

impl fmt::Display for CollapseStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //writeln!(f, "  Collapse Stats:")?;
        writeln!(f, "    Collapses Succeeded: {}", self.n_collapses)?;
        writeln!(f, "    Collapses Failed:    {}", self.n_fails)?;
        Ok(())
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SmoothStats {
    n_smooths: Idx,
    n_fails: Idx,
    n_verifs: Idx,
    r_stats: RemesherStats,
    total_verif_time: f64,
    total_smooth_time: f64,
    total_fails_time: f64,
}

impl SmoothStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>>(
        n_smooths: Idx,
        n_fails: Idx,
        n_verifs: Idx,
        r: &Remesher<D, E, M>,
        total_verif_time: f64,
        total_smooth_time: f64,
        total_fails_time: f64,
    ) -> Self {
        Self {
            n_smooths,
            n_fails,
            n_verifs,
            r_stats: RemesherStats::new(r),
            total_verif_time,
            total_smooth_time,
            total_fails_time,
        }
    }

    pub const fn get_n_smooths(&self) -> Idx {
        self.n_smooths
    }

    pub const fn get_n_fails(&self) -> Idx {
        self.n_fails
    }

    pub const fn get_n_verifs(&self) -> Idx {
        self.n_verifs
    }

    pub const fn get_time_smooth(&self) -> f64 {
        if self.n_smooths == 0 {
            0.0
        } else {
            self.total_smooth_time / self.n_smooths as f64
        }
    }

    pub const fn get_time_fails(&self) -> f64 {
        if self.n_fails == 0 {
            0.0
        } else {
            self.total_fails_time / self.n_fails as f64
        }
    }

    pub const fn get_time_verif(&self) -> f64 {
        if self.n_verifs == 0 {
            0.0
        } else {
            self.total_verif_time / self.n_verifs as f64
        }
    }
}
impl fmt::Display for SmoothStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //writeln!(f, "  Smooth Stats:")?;
        writeln!(f, "    Smooth Fails: {}", self.n_fails)?;
        Ok(())
    }
}
