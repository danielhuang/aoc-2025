use aoc_2025::*;

fn main() {
    let input = load_input();

    let red = input
        .lines()
        .map(|x| x.ints().ca())
        .map(|x| Cell(x) as Cell2)
        .cv();

    let red2 = red.cii().cset();

    let mut green = HashSet::new();

    for (a, b) in red.cii().circular_tuple_windows() {
        green.extend(a.goto(b));
    }

    let outside = bounds(red.cii());
    let outside: Cell2 = red
        .cii()
        .find_map(|corner| {
            corner
                .adj()
                .ii()
                .find(|&adj| outside.grow().contains(adj) && !outside.contains(adj))
        })
        .unwrap();

    let outside = bfs_reach([outside], |prev| {
        prev.adj().ii().filter(|x| {
            x.adj_diag()
                .ii()
                .any(|y| red2.contains(&y) || green.contains(&y))
                && !green.contains(x)
        })
    })
    .map(|x| x.0)
    .cset();

    let mut cands1 = vec![];
    let mut cands2 = vec![];

    for (a, b) in red.cii().tuple_combinations() {
        let b = bounds([a, b]);
        let area = b.volume();

        cands1.push(area);

        if outside.iter().all(|x| !b.contains(*x)) {
            cands2.push(area);
        }
    }

    cp(cands1.max_c());
    cp(cands2.max_c());
}
