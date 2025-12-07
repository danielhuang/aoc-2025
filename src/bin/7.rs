use aoc_2025::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x, '.');

    let b = bounds(grid.findv('^').ii().chain(grid.findv('S'))).grow();

    let beam = bfs_reach([grid.findv('S').one()], |prev| {
        let mut result = vec![];

        if grid[prev] == 'S' || grid[prev] == '.' {
            result.push(prev.down(1));
        } else {
            result.push(prev.left(1));
            result.push(prev.right(1));
        }

        result.ii().filter(|&x| b.contains(x))
    })
    .cv();

    cp(beam.ii().map(|x| x.0).filter(|x| grid[x] == '^').count());

    cp(count_paths(
        [grid.findv('S').one()],
        |prev| {
            let mut result = vec![];

            if grid[prev] == 'S' || grid[prev] == '.' {
                result.push(prev.down(1));
            } else {
                result.push(prev.left(1));
                result.push(prev.right(1));
            }

            result
        },
        |x| !b.contains(*x),
    ))
}
