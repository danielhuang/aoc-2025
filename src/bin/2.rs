use aoc_2025::*;

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut count = 0;

        for segment in input.split(',') {
            let [a, b] = segment.uints().ca();
            for id in a..=b {
                let id = id.to_string();
                for factor in 1..id.len() {
                    let accept = if part2 {
                        id.len() % factor == 0
                    } else {
                        id.len() == factor * 2
                    };
                    if accept && id.chars().chunks(factor).ii().map(|x| x.cv()).cset().len() == 1 {
                        count += id.int();
                        break;
                    }
                }
            }
        }

        cp(count);
    }
}
