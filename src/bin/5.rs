use aoc_2025::*;

fn main() {
    let input = load_input();

    let (head, body) = input.split_once("\n\n").unwrap();

    let mut map = Intervals::default();

    for [a, b] in head.lines().map(|x| x.uints().ca()) {
        map.add(a.int(), b.int() + 1);
    }

    let mut count = 0;

    for n in body.lines() {
        if map.contains(n.int()) {
            count += 1;
        }
    }

    cp(count);
    cp(map.covered_size());
}
