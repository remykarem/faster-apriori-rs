use std::collections::HashSet;

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet};
use pyo3::wrap_pyfunction;
type ItemId = usize;
type Itemset = Vec<usize>;
use ahash::AHashMap;
use pyo3::types::IntoPyDict;
use rayon::prelude::*;
use std::cmp::Ordering::{Equal, Greater};

fn main() {
    #[pymodule]
    fn apriori(_: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(generate_frequent_itemsets, m)?)?;
        Ok(())
    }
}

#[pyfunction]
#[pyo3(text_signature = "(transactions, min_support, max_length, /)")]
fn generate_frequent_itemsets(
    raw_transactions: Vec<HashSet<&str>>,
    min_support: f32,
    max_length: usize,
) -> Py<PyDict> {
    let itemset_counts = generate_frequent_itemsetsa(raw_transactions, min_support, max_length);

    convert_itemset_counts(itemset_counts)
}

macro_rules! pyfrozenset {
    ($py:expr,$x:expr) => {{
        let set: Py<PyFrozenSet> = PyFrozenSet::new($py, &$x).unwrap().into();
        set
    }};
}

pub fn convert_itemset_counts(
    itemset_counts: AHashMap<usize, AHashMap<Itemset, u32>>,
) -> Py<PyDict> {
    Python::with_gil(|py| {
        itemset_counts
            .into_iter()
            .map(|(size, itemset_counts)| {
                let py_itemset_counts: Py<PyDict> = itemset_counts
                    .into_iter()
                    .map(|(itemset, count)| (pyfrozenset![py, itemset], count))
                    .collect::<Vec<(Py<PyFrozenSet>, u32)>>()
                    .into_py_dict(py)
                    .into();
                (size, py_itemset_counts)
            })
            .into_py_dict(py)
            .into()
    })
}

struct TransactionManager<'a> {
    index: AHashMap<usize, Vec<usize>>,
    inventory: AHashMap<ItemId, &'a str>,
    n: usize,
}

impl<'a> TransactionManager<'a> {
    fn from(transactions: Vec<HashSet<&'a str>>) -> Self {
        let mut index = AHashMap::with_capacity(1024);
        let mut reverse_lookup = AHashMap::with_capacity(1024);
        let mut last_item_id = 0;

        transactions
            .iter()
            .enumerate()
            .for_each(|(txn_id, transactions)| {
                transactions.iter().for_each(|&item| {
                    let item_id = reverse_lookup.entry(item).or_insert_with(|| {
                        last_item_id += 1;
                        last_item_id
                    });

                    index
                        .entry(*item_id)
                        .or_insert_with(|| Vec::with_capacity(32))
                        .push(txn_id);
                });
            });

        let inventory = reverse_lookup.into_iter().map(|(k, v)| (v, k)).collect();
        let n = index.len();
        TransactionManager {
            index,
            inventory,
            n,
        }
    }
    fn calculate_support(&self, itemset: &mut [usize], min_support_count: usize) -> usize {
        if itemset.len() == 1 {
            return self.index.get(itemset.get(0).unwrap()).unwrap().len();
        } else if itemset.len() == 2 {
            let a = &mut self.index.get(itemset.get(0).unwrap()).unwrap().iter();
            let b = &mut self.index.get(itemset.get(1).unwrap()).unwrap().iter();
            let g = vec![a, b];
            count_intersection_len(g)
        } else if itemset.len() == 3 {
            let a = &mut self.index.get(itemset.get(0).unwrap()).unwrap().iter();
            let b = &mut self.index.get(itemset.get(1).unwrap()).unwrap().iter();
            let c = &mut self.index.get(itemset.get(2).unwrap()).unwrap().iter();
            let g = vec![a, b, c];
            count_intersection_len(g)
        } else if itemset.len() == 4 {
            let a = &mut self.index.get(itemset.get(0).unwrap()).unwrap().iter();
            let b = &mut self.index.get(itemset.get(1).unwrap()).unwrap().iter();
            let c = &mut self.index.get(itemset.get(2).unwrap()).unwrap().iter();
            let d = &mut self.index.get(itemset.get(3).unwrap()).unwrap().iter();
            let g = vec![a, b, c, d];
            count_intersection_len(g)
        } else {
            panic!("oops");
        }
    }
    fn create_initial_candidates(&self) -> Vec<Vec<usize>> {
        self.index.keys().into_iter().map(|&k| vec![k]).collect()
    }

    fn generate_counts(
        &self,
        candidates: Vec<Vec<usize>>,
        min_support_count: usize,
    ) -> AHashMap<Vec<usize>, u32> {
        let g : Vec<(Vec<usize>, u32)> = candidates
            .into_par_iter()
            .filter_map(|mut candidate| {
                let count = self.calculate_support(&mut candidate, min_support_count);
                if count >= min_support_count {
                    Some((candidate, count as u32))
                } else {
                    None
                }
            })
            .collect();

        println!("{}", g.len());
        
        g.into_iter().collect()
    }
}

/// Adapted from https://stackoverflow.com/questions/56261476/why-is-finding-the-intersection-of-integer-sets-faster-with-a-vec-compared-to-bt
fn intersection_sorted_vectors(xs: &[usize], ys: &[usize]) -> Vec<usize> {
    let mut want = Vec::with_capacity(8);

    let (xs, ys) = if xs.len() <= ys.len() {
        (xs, ys)
    } else {
        (ys, xs)
    };

    let mut xs_iter = xs.iter();
    if let Some(mut x) = xs_iter.next() {
        for y in ys {
            while x < y {
                x = match xs_iter.next() {
                    Some(x) => x,
                    None => return want,
                };
            }

            if x == y {
                want.push(*x);
            }
        }
    }

    want
}

/// Join k length itemsets into k + 1 length itemsets.
///
/// Algorithm translated from
/// https://github.com/tommyod/Efficient-Apriori/blob/master/efficient_apriori/itemsets.py
pub fn generate_candidates_k(counts: &AHashMap<Itemset, u32>) -> Vec<Itemset> {
    let mut itemsets: Vec<Itemset> = counts.keys().map(|s| s.to_owned()).collect();

    if itemsets.is_empty() {
        return vec![];
    }

    itemsets.sort_unstable();

    let mut final_itemsets: Vec<Itemset> = Vec::with_capacity(1024); // arbitrary
    let mut itemset_first_tuple: Itemset = Vec::with_capacity(itemsets[0].len() + 1);
    let mut tail_items: Vec<ItemId> = Vec::with_capacity(itemsets.len()); // based on analysis of the first for loop

    let mut i = 0;
    while i < itemsets.len() {
        let mut skip = 1;

        let (itemset_first, itemset_last) = itemsets[i].split_at(itemsets[i].len() - 1);
        let itemset_last = itemset_last.to_owned().pop().unwrap();

        tail_items.clear();
        tail_items.push(itemset_last);

        for j in (i + 1)..itemsets.len() {
            let (itemset_n_first, itemset_n_last) = itemsets[j].split_at(itemsets[j].len() - 1);
            let itemset_n_last = itemset_n_last.to_owned().pop().unwrap();

            if itemset_first == itemset_n_first {
                tail_items.push(itemset_n_last);
                skip += 1;
            } else {
                break;
            }
        }

        for combi in tail_items.iter().combinations(2).sorted() {
            itemset_first_tuple.clear();
            itemset_first_tuple.extend(itemset_first);
            let (a, b) = combi.split_at(1);
            let a = *a.to_owned().pop().unwrap();
            let b = *b.to_owned().pop().unwrap();

            itemset_first_tuple.push(a);
            itemset_first_tuple.push(b);
            // itemset_first_tuple.sort_unstable();
            final_itemsets.push(itemset_first_tuple.to_owned());
        }

        i += skip;
    }

    final_itemsets
}

fn generate_frequent_itemsetsa(
    transactions: Vec<HashSet<&str>>,
    min_support: f32,
    max_length: usize,
) -> AHashMap<usize, AHashMap<Itemset, u32>> {
    let mut tm = TransactionManager::from(transactions);
    let mut map = AHashMap::new();
    let min_support_count = (min_support * tm.index.len() as f32).ceil() as usize;

    println!("tm.index={}", tm.index.len());
    tm.index.retain(|_, v| v.len() >= min_support_count);
    println!("tm.index={}", tm.index.len());

    println!("Length 1");
    println!("  Generating candidates");
    let mut candidates = tm.create_initial_candidates();
    println!("  {} candidates found", candidates.len());

    for k in 1..=max_length {
        println!("  Generating counts");
        let counts = tm.generate_counts(candidates, min_support_count);
        println!("Length {}", k + 1);
        println!("  Generating candidates");
        candidates = generate_candidates(&counts, k + 1, max_length);
        println!("  {} candidates found", candidates.len());

        map.insert(k, counts);
    }

    println!("Rust done");

    map
}

fn generate_candidates(counts: &AHashMap<Vec<usize>, u32>, len: usize, max: usize) -> Vec<Itemset> {
    if len > max {
        vec![]
    } else if len == 2 {
        generate_candidates_2(counts)
    } else {
        generate_candidates_k(counts)
    }
}

fn generate_candidates_2(counts: &AHashMap<Vec<usize>, u32>) -> Vec<Itemset> {
    counts
        .keys()
        .flat_map(|s| s.iter())
        .combinations(2)
        .into_iter()
        .map(|c| c.into_iter().copied().sorted().collect::<Vec<usize>>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_jd() {
        assert_eq!(
            intersection_sorted_vectors(&[1, 2, 3], &[1, 2, 3]),
            vec![1, 2, 3]
        );
        assert_eq!(intersection_sorted_vectors(&[1], &[1, 2, 3]), vec![1]);
        assert_eq!(intersection_sorted_vectors(&[1, 2, 3], &[1]), vec![1]);
        assert_eq!(intersection_sorted_vectors(&[], &[1, 2, 3]), vec![]);
        assert_eq!(intersection_sorted_vectors(&[], &[]), vec![]);
    }

    #[test]
    fn test_count_intersection_len() {
        let a = &[1];
        let b = &[2];
        let c = &[3];

        let mut a = a.iter();
        let mut b = b.iter();
        let mut c = c.iter();

        let iterators = vec![&mut a, &mut b, &mut c];
        assert_eq!(count_intersection_len(iterators), 0);
    }
    #[test]
    fn test_yo3() {
        let a = &[1];
        let b = &[1];
        let c = &[1];

        let mut a = a.iter();
        let mut b = b.iter();
        let mut c = c.iter();

        let iterators = vec![&mut a, &mut b, &mut c];
        assert_eq!(count_intersection_len(iterators), 1);
    }
    #[test]
    fn test_yo4() {
        let a = &[1];
        let b = &[1, 99, 999, 999];
        let c = &[1];

        let mut a = a.iter();
        let mut b = b.iter();
        let mut c = c.iter();

        let iterators = vec![&mut a, &mut b, &mut c];
        assert_eq!(count_intersection_len(iterators), 1);
    }

    #[test]
    fn test_yo2() {
        let a = &[4, 10, 18, 19, 20, 21, 47];
        let b = &[4, 5, 7, 10, 20, 30, 31, 47, 50];
        let c = &[4, 16, 17, 18, 20, 30, 47, 51];

        let mut a = a.iter();
        let mut b = b.iter();
        let mut c = c.iter();

        let iterators = vec![&mut a, &mut b, &mut c];
        assert_eq!(count_intersection_len(iterators), 3);
    }
}

// Space: O(1)
// Time: O(hn)
fn count_intersection_len(mut iterators: Vec<&mut std::slice::Iter<usize>>) -> usize {
    let mut max = 0;
    let mut count = 0;
    let mut support_count = 0;
    let n = iterators.len();

    'mainloop: loop {
        for iter in iterators.iter_mut() {
            // If all iterators have the same max, add to support count
            // Then reset the counter
            if count == n {
                support_count += 1;
                count = 0;
            }

            // Stop if any iterator is empty
            if iter.len() == 0 {
                break 'mainloop;
            }

            for x in iter {
                match x.cmp(&max) {
                    Equal => {
                        count += 1;
                        break;
                    }
                    Greater => {
                        max = *x;
                        count = 1;
                        break;
                    }
                    _ => continue,
                }
            }
        }
    }
    support_count
}
