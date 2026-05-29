use rand::{Rng, RngExt, distr::{weighted::WeightedIndex, uniform::SampleUniform}, seq::IteratorRandom};
use rand_distr::Distribution;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const DEFAULT_CAPACITY: usize = 1000;

pub trait Float:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::fmt::Debug
    + std::fmt::Display
    + std::ops::Mul<Output = Self>
    + rand::distr::weighted::Weight  // needed for WeightedIndex
    + SampleUniform                  // needed for WeightedIndex struct bound
    + 'static
{
    /// `ceil(log2(self))` computed via bit manipulation.
    fn log2_ceil_bits(self) -> usize;

    /// `2^exp` as `Self`.
    fn two_pow(exp: usize) -> Self;

    /// Sample a uniform value in `[0, 1)`.
    fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl Float for f32 {
    #[inline] fn log2_ceil_bits(self) -> usize { log2_ceil2_f32(self) }
    #[inline] fn two_pow(exp: usize) -> Self    { 2.0f32.powi(exp as i32) }
    #[inline] fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> Self { rng.random::<f32>() }
}

impl Float for f64 {
    #[inline] fn log2_ceil_bits(self) -> usize { log2_ceil2_f64(self) }
    #[inline] fn two_pow(exp: usize) -> Self    { 2.0f64.powi(exp as i32) }
    #[inline] fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> Self { rng.random::<f64>() }
}

// ─── Slot ─────────────────────────────────────────────────────────────────────

/// Fused per-element data: weight, level, and index within the level bucket.
/// f32 variant: 12 bytes (4-aligned). f64 variant: 16 bytes (8-aligned, power-of-2).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct Slot<W> {
    weight: W,
    idx_in_level: u32,
    level: u8,
    _pad: [u8; 3],
}

impl<W: Float> Default for Slot<W> {
    fn default() -> Self {
        // W::default() == 0.0 for both f32 and f64
        Self { weight: W::default(), idx_in_level: 0, level: 0, _pad: [0; 3] }
    }
}

// ─── DynamicWeightedSampler ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DynamicWeightedSampler<W: Float = f32> {
    max_value: W,
    n_levels: usize,
    total_weight: W,
    slots: Vec<Slot<W>>,
    level_weight: Vec<W>,
    level_bucket: Vec<Vec<u32>>, // u32 instead of usize: 2× cache density
    level_max: Vec<W>,
}

impl<W: Float> DynamicWeightedSampler<W> {
    pub fn new(max_value: W) -> Self {
        Self::new_with_capacity(max_value, DEFAULT_CAPACITY)
    }

    pub fn new_with_capacity(max_value: W, physical_capacity: usize) -> Self {
        assert!(physical_capacity > 0);
        let n_levels = max_value.log2_ceil_bits() + 1;
        let max_value = W::two_pow(max_value.log2_ceil_bits());
        let slots = vec![Slot::default(); physical_capacity];
        let level_weight = vec![W::default(); n_levels];
        let level_bucket = vec![vec![]; n_levels];
        let top_level = n_levels - 1;
        let level_max: Vec<W> = (0..n_levels).map(|i| W::two_pow(top_level - i)).collect();
        Self {
            max_value,
            n_levels,
            total_weight: W::default(),
            slots,
            level_weight,
            level_bucket,
            level_max,
        }
    }

    pub fn insert(&mut self, id: usize, weight: W) {
        assert!(weight > W::default());
        if id > self.slots.len() - 1 {
            self.slots.resize(id + 1, Slot::default());
        }
        assert!(self.slots[id].weight == W::default(), "Inserting element id {id} with weight {weight}, but it already existed with weight {}", self.slots[id].weight);
        assert!(weight <= self.max_value, "Adding element {id} with weight {weight} exceeds the maximum weight capacity of {}", self.max_value);
        let level = self.level(weight);
        self.slots[id].weight = weight;
        self.total_weight += weight;
        self.insert_to_level(id, level, weight);
    }

    #[inline]
    fn level(&self, weight: W) -> usize {
        debug_assert!(weight <= self.max_value, "{weight} > {}", self.max_value);
        debug_assert!(weight > W::default());
        let top_level = self.n_levels - 1;
        let level_from_top = weight.log2_ceil_bits();
        debug_assert!(top_level >= level_from_top);
        top_level - level_from_top
    }

    #[inline]
    fn insert_to_level(&mut self, id: usize, level: usize, weight: W) {
        self.level_weight[level] += weight;
        let idx = self.level_bucket[level].len() as u32;
        self.level_bucket[level].push(id as u32);
        self.slots[id].idx_in_level = idx;
        self.slots[id].level = level as u8;
    }

    #[inline]
    fn remove_from_level(&mut self, id: usize, level: usize, weight: W) {
        debug_assert_eq!(self.level_bucket[level][self.slots[id].idx_in_level as usize] as usize, id);
        self.level_weight[level] -= weight;
        let idx_in_level = self.slots[id].idx_in_level as usize;
        let last_idx_in_level = self.level_bucket[level].len() - 1;
        if idx_in_level != last_idx_in_level {
            let id_in_last = self.level_bucket[level][last_idx_in_level] as usize;
            self.level_bucket[level].swap(idx_in_level, last_idx_in_level);
            self.slots[id_in_last].idx_in_level = idx_in_level as u32;
        }
        self.level_bucket[level].pop();
        self.slots[id].idx_in_level = 0;
        self.slots[id].level = 0;
    }

    pub fn remove(&mut self, id: usize) -> W {
        let slot = self.slots[id]; // single cache-line fetch: weight + level + idx_in_level
        debug_assert!(slot.weight > W::default(), "removing element {id} with 0 weight");
        self.slots[id].weight = W::default();
        self.total_weight -= slot.weight;
        self.remove_from_level(id, slot.level as usize, slot.weight);
        slot.weight
    }

    pub fn update(&mut self, id: usize, new_weight: W) {
        let slot = self.slots[id]; // single read
        if slot.weight == new_weight {
            return;
        }
        if new_weight == W::default() {
            // inline remove to avoid re-reading slots[id]
            debug_assert!(slot.weight > W::default(), "removing element {id} with 0 weight");
            self.slots[id].weight = W::default();
            self.total_weight -= slot.weight;
            self.remove_from_level(id, slot.level as usize, slot.weight);
            return;
        }
        if slot.weight == W::default() {
            self.insert(id, new_weight);
            return;
        }
        let new_level = self.level(new_weight);
        self.total_weight += new_weight;
        self.total_weight -= slot.weight;
        self.slots[id].weight = new_weight;
        if slot.level as usize == new_level {
            // inline update_weight_in_level to avoid function call overhead
            self.level_weight[slot.level as usize] += new_weight;
            self.level_weight[slot.level as usize] -= slot.weight;
        } else {
            self.remove_from_level(id, slot.level as usize, slot.weight);
            self.insert_to_level(id, new_level, new_weight);
        }
    }

    pub fn update_delta(&mut self, id: usize, delta: W) {
        let slot = self.slots[id]; // single read
        let new_weight = slot.weight + delta;
        if new_weight == slot.weight {
            return;
        }
        if new_weight <= W::default() {
            if slot.weight > W::default() {
                self.slots[id].weight = W::default();
                self.total_weight -= slot.weight;
                self.remove_from_level(id, slot.level as usize, slot.weight);
            }
            return;
        }
        if slot.weight == W::default() {
            self.insert(id, new_weight);
            return;
        }
        let new_level = self.level(new_weight);
        self.total_weight += delta;
        self.slots[id].weight = new_weight;
        if slot.level as usize == new_level {
            self.level_weight[slot.level as usize] += delta;
        } else {
            self.remove_from_level(id, slot.level as usize, slot.weight);
            self.insert_to_level(id, new_level, new_weight);
        }
    }

    #[inline]
    pub fn get_weight(&self, id: usize) -> W {
        self.slots[id].weight
    }

    #[inline]
    pub fn get_total_weight(&self) -> W {
        self.total_weight
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        assert!(self.total_weight <= self.max_value, "weighted sampler total weight {} is bigger than max weight {}.", self.total_weight, self.max_value);
        let levels_sampler = WeightedIndex::new(self.level_weight.iter().copied()).ok()?;
        let level = levels_sampler.sample(rng);

        loop {
            let idx_in_level = (0..self.level_bucket[level].len()).choose(rng).unwrap();
            let sampled_id = self.level_bucket[level][idx_in_level] as usize;
            let weight = self.slots[sampled_id].weight;
            // Split into two asserts to avoid type-inference confusion between W and usize.
            debug_assert!(weight <= self.level_max[level]);
            debug_assert!(level == self.n_levels - 1 || self.level_max[level + 1] < weight);
            let u = W::random_unit(rng) * self.level_max[level];
            if u <= weight {
                break Some(sampled_id);
            }
        }
    }

    pub fn check_invariant(&self) -> bool {
        let sum_level = self.level_weight.iter().fold(W::default(), |acc, &w| acc + w);
        let sum_slots = self.slots.iter().fold(W::default(), |acc, s| acc + s.weight);
        sum_level == self.total_weight
            && sum_slots == self.total_weight
            && self.total_weight <= self.max_value
    }
}

// ─── log2_ceil helpers ────────────────────────────────────────────────────────

fn log2_ceil2_f32(weight: f32) -> usize {
    let b: u32 = weight.to_bits();
    let e = (b >> 23) & 0xFF;        // 8-bit exponent field
    let frac = b & ((1 << 23) - 1); // 23-bit mantissa
    let z = if frac == 0 { e as i32 - 127 } else { e as i32 - 126 };
    z as usize
}

fn log2_ceil2_f64(weight: f64) -> usize {
    let b: u64 = weight.to_bits();
    let e = (b >> 52) & ((1 << 11) - 1);
    let frac = b & ((1 << 52) - 1);
    let z = if frac == 0 { e as i64 - 1023 } else { e as i64 - 1022 };
    z as usize
}

#[cfg(test)]
mod test_weighted_sampler {
    use std::time::Instant;
    use std::collections::HashMap;
    use rand::rng;
    use super::*;

    #[test]
    fn test_distr() {
        let mut sampler = DynamicWeightedSampler::new_with_capacity(1000., 5);
        let mut samples: HashMap<usize, usize> = HashMap::new();

        sampler.insert(1, 999.);
        sampler.insert(2, 1.);

        let n_samples = 1_000_000;
        let start = Instant::now();
        for _ in 1..n_samples {
            let sample = sampler.sample(&mut rng()).unwrap();
            *samples.entry(sample).or_default() += 1;
        }
        let duration = start.elapsed();

        assert!(duration.as_secs() <= 3);
        approx::assert_abs_diff_eq!(samples[&1] as f32 / n_samples as f32, 0.999, epsilon = 1e-4);
        approx::assert_abs_diff_eq!(samples[&2] as f32 / n_samples as f32, 0.001, epsilon = 1e-4);

        sampler.update(1, 99.);

        samples.drain();
        let n_samples = 1_000;
        for _ in 1..n_samples {
            let sample = sampler.sample(&mut rng()).unwrap();
            *samples.entry(sample).or_default() += 1;
        }

        approx::assert_abs_diff_eq!(samples[&1] as f32 / n_samples as f32, 0.99, epsilon = 1e-2);
        approx::assert_abs_diff_eq!(samples[&2] as f32 / n_samples as f32, 0.01, epsilon = 1e-2);
    }

    #[test]
    fn test_distr_f64() {
        let mut sampler = DynamicWeightedSampler::<f64>::new_with_capacity(1000., 5);
        let mut samples: HashMap<usize, usize> = HashMap::new();

        sampler.insert(1, 999.);
        sampler.insert(2, 1.);

        let n_samples = 1_000_000;
        for _ in 1..n_samples {
            let sample = sampler.sample(&mut rng()).unwrap();
            *samples.entry(sample).or_default() += 1;
        }

        approx::assert_abs_diff_eq!(samples[&1] as f64 / n_samples as f64, 0.999, epsilon = 1e-4);
        approx::assert_abs_diff_eq!(samples[&2] as f64 / n_samples as f64, 0.001, epsilon = 1e-4);
    }

    #[test]
    fn test_remove() {
        let mut sampler = DynamicWeightedSampler::new_with_capacity(1000., 5);
        let level = sampler.level(500.);
        sampler.insert(1, 500.);
        assert_eq!(Some(&1u32), sampler.level_bucket[level].get(0));
        sampler.insert(2, 510.);
        assert_eq!(Some(&2u32), sampler.level_bucket[level].get(1));
        sampler.remove(1);
        assert_eq!(Some(&2u32), sampler.level_bucket[level].get(0));
        sampler.insert(1, 500.);
        assert_eq!(Some(&1u32), sampler.level_bucket[level].get(1));
        sampler.remove(1);
    }

    #[test]
    fn test_level() {
        let sampler = DynamicWeightedSampler::new_with_capacity(1000., 5);
        assert_eq!(11, sampler.n_levels);
        assert_eq!(11 - 1, sampler.level(1.));
        assert_eq!(11 - 2, sampler.level(2.));
        assert_eq!(11 - 3, sampler.level(3.));
        assert_eq!(11 - 3, sampler.level(4.));
    }
}
