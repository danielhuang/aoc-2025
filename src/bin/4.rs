use aoc_2025::*;

fn main() {
    let input = load_input();

    let mut grid = parse_grid(&input, |x| x, '.');

    let orig_paper_count = grid.findv('@').len();

    let mut part1 = 0;

    loop {
        let mut changed = false;

        let mut to_remove = vec![];

        let papers = grid.findv('@');

        for paper in papers {
            if paper.adj_diag().ii().filter(|x| grid[*x] == '@').count() < 4 {
                to_remove.push(paper);
            }
        }

        if part1 == 0 {
            part1 = to_remove.len();
        }

        for paper in to_remove {
            grid[paper] = '.';
            changed = true;
        }

        if !changed {
            break;
        }
    }

    cp(part1);
    cp(orig_paper_count - grid.findv('@').len());
}
