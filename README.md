# dynamic_weighted_sampler

A dynamically-updated, weighted random sampler for discrete items, based on the algorithm described by [Aaron Defazio](https://www.aarondefazio.com/tangentially/?p=58), itself derived from the method presented in:

> Yossi Matias, Jeffrey Scott Vitter, and Wen-Chun Ni.
> *Dynamic Generation of Discrete Random Variates*.
> **Theory of Computing Systems**, 36 (2003): 329–358.
> [Springer Link](https://link.springer.com/article/10.1007/s00224-003-1077-3)

This crate allows for efficient weighted sampling from a collection where item weights can be changed at runtime, and supports fast sampling and updates.

---

## ✨ Features

- Dynamic updates to weights
- Efficient close to constant sampling time
- Efficient constant weight update time
- Optional `serde` support via a feature flag

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
dynamic_weighted_sampler = "0.1"
```

To enable serialization support:

```toml
[dependencies]
dynamic_weighted_sampler = { version = "0.1", features = ["serde"] }
```

---

## 🔧 Usage

```rust
use dynamic_weighted_sampler::DynamicWeightedSampler;

let mut sampler = DynamicWeightedSampler::new();
sampler.insert(1, 4.);  // 80% of the mass
sampler.insert(2, 1.); // 20% of the mass

// Sample an item
let item = sampler.sample(&mut rand::rng());
println!("Sampled: {:?}", item);

// Update the weight
sampler.update(1, 5.);
```

---

## 🔒 License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to open a [pull request](https://github.com/germank/dynamic_weighted_sampler/pulls) or [issue](https://github.com/germank/dynamic_weighted_sampler/issues).

---

## 📈 Crate Info

[![Crates.io](https://img.shields.io/crates/v/dynamic_weighted_sampler.svg)](https://crates.io/crates/dynamic_weighted_sampler)
[![Docs.rs](https://docs.rs/dynamic_weighted_sampler/badge.svg)](https://docs.rs/dynamic_weighted_sampler)