from pathlib import Path

from z3 import *

# todo: don't do this

INPUT = Path("target/10.input.txt").read_text()

print(INPUT)

ans = 0

for line in INPUT.splitlines():
    _, *buttons, expected = line.split()
    buttons = [[int(x) for x in button[1:-1].split(",")] for button in buttons]
    expected = [int(x) for x in expected[1:-1].split(",")]

    opt = Optimize()

    total_presses = Int("total_presses")

    charges = []

    for i in range(len(expected)):
        charges.append(Int(f"charge_{i}"))

        incs = []

        for button_id, button in enumerate(buttons):
            button_count = Int(f"button_{button_id}")
            opt.add(button_count >= 0)
            for button_i in button:
                if button_i == i:
                    incs.append(button_count)

        print(incs)

        opt.add(Sum(incs) == expected[i])

    opt.minimize(total_presses)

    opt.add(
        total_presses
        == Sum([Int(f"button_{button_id}") for button_id in range(len(buttons))])
    )

    print(opt.check())
    print(opt.model())

    ans += int(str(opt.model()[Int("total_presses")]))

print(ans)
