use aoc_2025::*;

fn main() {
    let input = load_input();

    let grid = input
        .lines()
        .rev()
        .skip(1)
        .cii()
        .rev()
        .map(|x| x.ints())
        .cv();
    let grid = transpose_vec(grid);

    let ops = input.lines().last().unwrap().split_whitespace().cv();

    let mut sum = 0;

    for (nums, op) in grid.cii().zip(ops) {
        if op == "+" {
            sum += nums.cii().sumi();
        } else if op == "*" {
            sum += nums.cii().product::<Z>();
        } else {
            dbg!(&op);
            todo!()
        }
    }

    cp(sum);

    let grid = input.lines().map(|x| x.chars().cv()).cv();
    let grid = transpose_vec(grid);
    let lines: Vec<String> = grid.ii().map(|x| x.ii().collect()).cv();
    let lines = lines
        .ii()
        .map(|x| {
            if x.trim().is_empty() {
                "".to_string()
            } else {
                x
            }
        })
        .cv();

    let groups = lines.split(|x| x.is_empty()).cv();

    let mut sum = 0;
    for group in groups {
        let op = group
            .iter()
            .flat_map(|x| x.chars())
            .find(|&x| x == '*' || x == '+')
            .unwrap();
        let nums = group.ii().map(|x| x.ints().one());
        if op == '+' {
            sum += nums.sumi();
        } else {
            assert_eq!(op, '*');
            sum += nums.product::<Z>();
        }
    }
    cp(sum);
}
