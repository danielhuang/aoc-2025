use aoc_2025::*;

fn main() {
    let input = load_input();

    let mut count = 0;

    for line in input.lines() {
        let lights = line.split(' ').cv()[0]
            .replace(['[', ']'], "")
            .chars()
            .map(|x| x == '#')
            .cv();

        let buttons = line
            .split(' ')
            .skip(1)
            .cii()
            .rev()
            .cii()
            .skip(1)
            .cii()
            .map(|x| x.uints())
            .cv();

        let path = bfs(
            [lights],
            |prev| {
                let prev = prev.clone();
                buttons.cii().map(move |button| {
                    let mut lights = prev.clone();
                    for i in button {
                        lights[i] = !lights[i];
                    }
                    lights
                })
            },
            |x| x.iter().all(|&x| !x),
        )
        .unwrap()
        .cv();

        count += path.len() - 1;
    }

    cp(count);
}
