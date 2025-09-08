import os
import json


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


def count_call_instruction(instr, call_set, already_add):
    inst = instr.get("inst")
    if inst == "call":
        # recurse into ops
        if "ops" in instr:

            ops = instr["ops"]
            if len(ops) == 0:
                return
            callee = ops[0]['const']
            if callee not in already_add:
                call_set[callee] = call_set.get(callee, 0) + 1
                already_add.add(callee)
            for op in instr["ops"]:
                count_call_instruction(op, call_set, already_add)




def count_call_types(data, call_set):
    already_add = set()
    for path_idx, path in enumerate(data["paths"], 1):
        path_detail = path["detail"]
        for step_idx, step in enumerate(reversed(path_detail), 1):
            cond = step.get("condition")
            if cond:
                count_call_instruction(cond, call_set, already_add)

    return call_set


if __name__ == "__main__":

    call_sets = {}
    for item in os.listdir("extracted_smt"):
        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)
        count_call_types(data, call_sets)

    # Sort by values in descending order
    sorted_data = dict(sorted(call_sets.items(), key=lambda item: item[1], reverse=True))
    top_100_set = set()
    for item in sorted_data:
        top_100_set.add(item)
        print(item)
        if len(top_100_set) > 38:
            break


    # let's see how many
    ok = 0
    for item in os.listdir("extracted_smt"):
        call_sets = {}
        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)
        count_call_types(data, call_sets)
        remaining_set = set(call_sets.keys()) - top_100_set
        if len(remaining_set) == 0:
            ok += 1
    print(ok)
