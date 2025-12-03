use aoc_2025::*;

fn best(digits: &[Z], n: usize, cache: &mut FxHashMap<(usize, usize), Vec<Z>>) -> Vec<Z> {
    if n == digits.len() {
        digits.to_vec()
    } else if n == 0 {
        vec![]
    } else {
        if let Some(ans) = cache.get(&(digits.len(), n)) {
            return ans.clone();
        }

        let mut cands = vec![];

        for i in 0..=(digits.len() - n) {
            let mut partial = vec![digits[i]];
            if n > 0 {
                let rest = best(&digits[(i + 1)..], n - 1, cache);
                partial.extend(rest);
                cands.push(partial);
            }
        }
        let ans = cands.ii().max().unwrap();
        cache.insert((digits.len(), n), ans.clone());
        ans
    }
}

fn main() {
    let input = load_input();

    for n in [2, 12] {
        let mut count = 0;

        for line in input.lines() {
            let digits = line.chars().map(|x| x.int()).cv();
            count += best(&digits, n, &mut FxHashMap::default())
                .ii()
                .map(|x| x.to_string())
                .collect_string()
                .int();
        }

        cp(count);
    }
}
