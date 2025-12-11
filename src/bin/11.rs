use aoc_2025::*;

fn main() {
    let input = load_input();

    let routes: HashMap<_, _> = input
        .lines()
        .map(|x| x.split_once(": ").unwrap())
        .map(|(a, b)| (a.tos(), b.split_whitespace().map(|x| x.tos()).cv()))
        .collect();

    for part2 in [false, true] {
        cp(count_paths(
            [(if part2 { "svr".tos() } else { "you".tos() }, false, false)],
            |(x, dac, fft)| {
                if x == "out" {
                    return vec![];
                }
                let nexts = routes[x].clone();
                let dac = *dac || !part2;
                let fft = *fft || !part2;
                nexts
                    .cii()
                    .map(move |next| {
                        if next == "dac" {
                            (next.tos(), true, fft)
                        } else if next == "fft" {
                            (next.tos(), dac, true)
                        } else {
                            (next.tos(), dac, fft)
                        }
                    })
                    .cv()
            },
            |x| *x == ("out".tos(), true, true),
        ));
    }
}
