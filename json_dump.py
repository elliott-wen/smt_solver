
def print_instruction(instr, indent=0):
    pad = "  " * indent
    inst = instr.get("inst")
    const = instr.get("const")
    if not inst:
        raise Exception("JSON Malformed: No inst")

    if const is not None:
        print(f"{pad}- {inst}: {const}")
    else:
        print(f"{pad}- {inst}")

    # recurse into ops
    if "ops" in instr:
        for op in instr["ops"]:
            print_instruction(op, indent + 1)


def print_paths(data):
    for path_idx, path in enumerate(data["paths"], 1):
        path_detail = path["detail"]
        print(f"Path {path_idx}:")
        for step_idx, step in enumerate(reversed(path_detail), 1):
            cond = step.get("condition")

            print(
                f"  Step {step_idx} (isSwitch={step['isSwitch']}, taken={step['taken']}):")
            if cond:
                print_instruction(cond, indent=2)
        print()