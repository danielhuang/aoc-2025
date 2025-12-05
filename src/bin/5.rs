use aoc_2025::*;

fn main() {
    let input = load_input();

    let (head, body) = input.split_once("\n\n").unwrap();

    let mut map = Intervals::default();

    for [a, b] in head.lines().map(|x| x.uints().ca()) {
        map.add(a.int(), b.int() + 1);
    }

    cp(body.ints().ii().filter(|&n| map.contains(n)).count());
    cp(map.covered_size());
}
