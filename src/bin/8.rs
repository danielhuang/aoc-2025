use aoc_2025::*;

fn main() {
    let input = load_input();

    let cells = input
        .lines()
        .map(|x| x.ints().ca())
        .map(|x| Cell(x) as Cell3)
        .cv();

    let mut disjoint = DisjointSet::using(cells.clone());

    let mut iters = if DEBUG { 10 } else { 1000 };
    let mut circuits = cells.len();

    for (a, b) in cells
        .ii()
        .tuple_combinations()
        .sorted_by_key(|(a, b)| (*b - *a).dist_sq())
    {
        if disjoint.join(a, b) {
            circuits -= 1;
        }

        if circuits == 1 {
            cp(a[0] * b[0]);
            return;
        }

        iters -= 1;
        if iters == 0 {
            let sets = disjoint.sets().ii().sorted_by_key(|x| x.len()).rev().cv();
            cp(sets[0].len() * sets[1].len() * sets[2].len());
        }
    }
}
