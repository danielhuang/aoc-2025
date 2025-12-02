use aoc_2025::*;

fn main() {
    let input = load_input();

    let mut dial: Z = 50;

    let mut count2 = 0;
    let mut count1 = 0;

    for line in input.lines() {
        let num = line.ints()[0];
        for _ in 0..num {
            if line.starts_with('L') {
                dial -= 1;
            } else {
                dial += 1;
            }
            dial = dial.rem_euclid(100);
            if dial == 0 {
                count2 += 1;
            }
        }
        if dial == 0 {
            count1 += 1;
        }
    }

    cp(count1);
    cp(count2);
}
