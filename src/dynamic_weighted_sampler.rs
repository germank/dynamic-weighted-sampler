use rand::{distr::weighted::WeightedIndex, seq::IteratorRandom, Rng};
use rand_distr::Distribution;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sugars::cvec;

const DEFAULT_CAPACITY: usize = 1000;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DynamicWeightedSampler {
    max_value: f64,
    n_levels: usize,
    total_weight: f64,
    weights: Vec<f64>,
    level_weight: Vec<f64>,
    level_bucket: Vec<Vec<usize>>,
    rev_level_bucket: Vec<usize>, // maps id -> idx within a level
    level_max: Vec<f64>,
}

impl DynamicWeightedSampler {
    pub fn new(max_value: f64) -> Self {
        Self::new_with_capacity(max_value, DEFAULT_CAPACITY)
    }

    pub fn new_with_capacity(max_value: f64, physical_capacity: usize) -> Self {
        assert!(physical_capacity > 0);
        let n_levels = max_value.log2().ceil() as usize + 1;
        let max_value = 2f64.powf(max_value.log2().ceil());
        let total_weight = 0.;
        let weights = vec![0.; physical_capacity];
        let level_weight = vec![0.; n_levels];
        let level_bucket = vec![vec![]; n_levels];
        let rev_level_bucket = vec![0; physical_capacity];
        let top_level = n_levels - 1;
        let level_max = cvec![2usize.pow(top_level as u32 - i) as f64; i in 0u32..(n_levels as u32)];
        Self {
            max_value,
            n_levels,
            total_weight,
            weights,
            level_weight,
            level_bucket,
            rev_level_bucket,
            level_max,
        }
    }

    pub fn insert(&mut self, id: usize, weight: f64) {
        assert!(weight > 0.);
        if id > self.weights.len() - 1 {
            self.weights.resize(id + 1, 0.);
            self.rev_level_bucket.resize(id + 1, 0);
        }
        assert!(self.weights[id] == 0., "Inserting element id {id} with weight {weight}, but it already existed with weight {}", self.weights[id]);
        assert!(weight <= self.max_value, "Adding element {id} with weight {weight} exceeds the maximum weight capacity of {}", self.max_value);
        self.weights[id] = weight;
        self.total_weight += weight;
        let level = self.level(weight);
        self.insert_to_level(id, level, weight)
    }

    fn level(&self, weight: f64) -> usize {
        assert!(weight <= self.max_value, "{weight} > {}", self.max_value);
        assert!(weight > 0.);
        let top_level = self.n_levels - 1;
        let level_from_top = log2_ceil2(weight);
        assert!(top_level >= level_from_top);
        let level =  top_level - level_from_top;
        level
    }

    #[inline(always)]
    fn insert_to_level(&mut self, id: usize, level: usize, weight: f64) {
        self.level_weight[level] += weight;
        self.level_bucket[level].push(id);
        self.rev_level_bucket[id] = self.level_bucket[level].len() - 1;
    }

    #[inline(always)]
    fn remove_from_level(&mut self, id: usize, level: usize, weight: f64) {
        assert_eq!(self.level_bucket[level][self.rev_level_bucket[id]], id);
        self.level_weight[level] -= weight;
        let idx_in_level = self.rev_level_bucket[id];
        let last_idx_in_level = self.level_bucket[level].len() - 1;
        if idx_in_level != last_idx_in_level {
            // swap with last element
            let id_in_last_idx = self.level_bucket[level][last_idx_in_level];
            self.level_bucket[level].swap(idx_in_level, last_idx_in_level);
            self.rev_level_bucket[id_in_last_idx] = idx_in_level;
        }
        // idx is last, just remove
        self.level_bucket[level].pop();
        self.rev_level_bucket[id] = 0;
    }

    pub fn remove(&mut self, id: usize) -> f64 {
        assert!(self.weights[id] > 0., "removing element {id} with 0 weight");
        let weight = self.weights[id];
        self.weights[id] = 0.;
        self.total_weight -= weight;
        let level = self.level(weight);
        self.remove_from_level(id, level, weight);
        weight
    }

    pub fn update(&mut self, id: usize, new_weight: f64) {
        if self.get_weight(id) == new_weight {
            // nothing to do
            return;
        }
        if new_weight == 0. {
            // remove it completely if the weight is 0
            self.remove(id);
            return
        }
        let curr_weight = self.weights[id];
        if curr_weight == 0. {
            // if the previous weight was 0, just insert it
            self.insert(id, new_weight);
            return;
        }
        // otherwise update the weight
        let curr_level = self.level(curr_weight);
        let new_level = self.level(new_weight);
        // Update the weight at the global level
        self.total_weight += new_weight - curr_weight;
        self.weights[id] = new_weight;
        if curr_level == new_level {
            // If the level didn't change, just update the level's weight
            self.update_weight_in_level(curr_level, curr_weight, new_weight);
        } else {
            // Otherwise, remove the element from the current level (if any)
            if curr_weight > 0. {
                self.remove_from_level(id, curr_level, curr_weight);
            }
            // and insert it to the new level (if any)
            if new_weight > 0. {
                self.insert_to_level(id, new_level, new_weight);
            }
        }
    }

    pub fn update_delta(&mut self, id: usize, delta: f64) {
        let new_weight = self.weights.get(id).unwrap_or(&0.) + delta;
        self.update(id, new_weight);
    }

    #[inline(always)]
    fn update_weight_in_level(&mut self, level: usize, curr_weight: f64, new_weight: f64) {
        self.level_weight[level] += new_weight - curr_weight;
    }

    pub fn get_weight(&self, id: usize) -> f64 {
        self.weights[id]
    }

    pub fn get_total_weight(&self) -> f64 {
        self.total_weight
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        assert!(self.total_weight <= self.max_value, "weighted sampler total weight {} is bigger than max weight {}.", self.total_weight, self.max_value);
        let levels_sampler = WeightedIndex::new(self.level_weight.iter().copied()).unwrap();
        let level = levels_sampler.sample(rng);

        loop {
            let idx_in_level = (0..self.level_bucket[level].len()).choose(rng).unwrap();
            let sampled_id = self.level_bucket[level][idx_in_level];
            let weight = self.weights[sampled_id];
            debug_assert!(weight <= self.level_max[level] && (level == self.n_levels - 1 || (self.level_max[level+1] < weight )));
            let u = rng.random::<f64>() * self.level_max[level];
            if u <= weight {
                break sampled_id;
            }
        }
    }

    pub fn check_invariant(&self) -> bool {
        self.level_weight.iter().sum::<f64>() == self.total_weight &&
        self.weights.iter().sum::<f64>() == self.total_weight &&
            self.total_weight <= self.max_value
    }
}

fn log2_ceil2(weight: f64) -> usize {
    let b: u64 = weight.to_bits();
    // let s = (b >> 63) & 1;
    let e = (b >> 52) & ((1<<11)-1);
    let frac = b & ((1<<52) -1);
    let z = if frac==0 { e as i64 - 1023 } else { e as i64 -1022 };
    z as usize
}

fn _log2_ceil(weight: f64) -> usize {
    // Define a lookup table with the first 34 powers of two, starting from 1.0
    let lookup_table: [f64; 34] = [
        1.0,          // ceil(log2(weight)) == 0 for (0, 1.0]
        2.0,          // ceil(log2(weight)) == 1 for (1.0, 2.0]
        4.0,          // ceil(log2(weight)) == 2 for (2.0, 4.0]
        8.0,          // ceil(log2(weight)) == 3 for (4.0, 8.0]
        16.0,         // ceil(log2(weight)) == 4 for (8.0, 16.0]
        32.0,         // ceil(log2(weight)) == 5 for (16.0, 32.0]
        64.0,         // ceil(log2(weight)) == 6 for (32.0, 64.0]
        128.0,        // ceil(log2(weight)) == 7 for (64.0, 128.0]
        256.0,        // ceil(log2(weight)) == 8 for (128.0, 256.0]
        512.0,        // ceil(log2(weight)) == 9 for (256.0, 512.0]
        1024.0,       // ceil(log2(weight)) == 10 for (512.0, 1024.0]
        2048.0,       // ceil(log2(weight)) == 11 for (1024.0, 2048.0]
        4096.0,       // ceil(log2(weight)) == 12 for (2048.0, 4096.0]
        8192.0,       // ceil(log2(weight)) == 13 for (4096.0, 8192.0]
        16384.0,      // ceil(log2(weight)) == 14 for (8192.0, 16384.0]
        32768.0,      // ceil(log2(weight)) == 15 for (16384.0, 32768.0]
        65536.0,      // ceil(log2(weight)) == 16 for (32768.0, 65536.0]
        131072.0,     // ceil(log2(weight)) == 17 for (65536.0, 131072.0]
        262144.0,     // ceil(log2(weight)) == 18 for (131072.0, 262144.0]
        524288.0,     // ceil(log2(weight)) == 19 for (262144.0, 524288.0]
        1048576.0,    // ceil(log2(weight)) == 20 for (524288.0, 1048576.0]
        2097152.0,    // ceil(log2(weight)) == 21 for (1048576.0, 2097152.0]
        4194304.0,    // ceil(log2(weight)) == 22 for (2097152.0, 4194304.0]
        8388608.0,    // ceil(log2(weight)) == 23 for (4194304.0, 8388608.0]
        16777216.0,   // ceil(log2(weight)) == 24 for (8388608.0, 16777216.0]
        33554432.0,   // ceil(log2(weight)) == 25 for (16777216.0, 33554432.0]
        67108864.0,   // ceil(log2(weight)) == 26 for (33554432.0, 67108864.0]
        134217728.0,  // ceil(log2(weight)) == 27 for (67108864.0, 134217728.0]
        268435456.0,  // ceil(log2(weight)) == 28 for (134217728.0, 268435456.0]
        536870912.0,  // ceil(log2(weight)) == 29 for (268435456.0, 536870912.0]
        1073741824.0, // ceil(log2(weight)) == 30 for (536870912.0, 1073741824.0]
        2147483648.0, // ceil(log2(weight)) == 31 for (1073741824.0, 2147483648.0]
        4294967296.0, // ceil(log2(weight)) == 32 for (2147483648.0, 4294967296.0]
        8589934592.0, // ceil(log2(weight)) == 33 for (4294967296.0, 8589934592.0]
    ];

    // Use binary search to find the index in the lookup table.
    match lookup_table.binary_search_by(|&upper_bound| upper_bound.partial_cmp(&weight).unwrap()) {
        Ok(index) => index as usize,        // Exact match found
        Err(_) => weight.log2().ceil() as usize,       // No match, but `Err` gives the insertion point
    }
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
            let sample = sampler.sample(&mut rng());
            *samples.entry(sample).or_default() += 1;
        }
        let duration = start.elapsed();

        assert!(duration.as_secs() <= 3); // 2-3 microseconds per sample
        approx::assert_abs_diff_eq!(samples[&1] as f64 / n_samples as f64, 0.999, epsilon=1e-4);
        approx::assert_abs_diff_eq!(samples[&2] as f64 / n_samples as f64, 0.001, epsilon=1e-4);

        println!("{:?}", sampler);
        sampler.update(1, 99.);
        println!("{:?}", sampler);

        samples.drain();
        let n_samples = 1_000;
        for _ in 1..n_samples {
            let sample = sampler.sample(&mut rng());
            *samples.entry(sample).or_default() += 1;
        }

        approx::assert_abs_diff_eq!(samples[&1] as f64 / n_samples as f64, 0.99, epsilon=1e-2);
        approx::assert_abs_diff_eq!(samples[&2] as f64 / n_samples as f64, 0.01, epsilon=1e-2);
    }

    #[test]
    fn test_remove() {
        let mut sampler = DynamicWeightedSampler::new_with_capacity(1000., 5);
        let level = sampler.level(500.);
        sampler.insert(1, 500.);
        assert_eq!(Some(&1), sampler.level_bucket[level].get(0));
        sampler.insert(2, 510.);
        assert_eq!(Some(&2), sampler.level_bucket[level].get(1));
        sampler.remove(1);
        assert_eq!(Some(&2), sampler.level_bucket[level].get(0));
        sampler.insert(1, 500.);
        assert_eq!(Some(&1), sampler.level_bucket[level].get(1));
        sampler.remove(1);
    }

    #[test]
    fn test_level() {
        let sampler = DynamicWeightedSampler::new_with_capacity(1000., 5);
        assert_eq!(11, sampler.n_levels);
        assert_eq!(11-1, sampler.level(1.));
        assert_eq!(11-2, sampler.level(2.));
        assert_eq!(11-3, sampler.level(3.));
        assert_eq!(11-3, sampler.level(4.));
    }
}