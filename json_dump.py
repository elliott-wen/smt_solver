import os
import json
import pickle


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


def count_call_instruction(instr, already_add):
    inst = instr.get("inst")
    if "ops" in instr:
        for op in instr["ops"]:
            count_call_instruction(op, already_add)

    if inst == "call":
        # recurse into ops
        if "ops" in instr:
            ops = instr["ops"]
            if len(ops) == 0:
                return
            callee = ops[0]['const']
            if callee not in already_add:
                already_add.add(callee)


def count_call_types(data):
    already_add = set()
    for path_idx, path in enumerate(data["paths"], 1):
        path_detail = path["detail"]
        for step_idx, step in enumerate(reversed(path_detail), 1):
            cond = step.get("condition")
            if cond:
                count_call_instruction(cond, already_add)

    return already_add


def get_popular_call():
    call_sets = {}
    for item in os.listdir("extracted_smt"):
        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)

        already_add = count_call_types(data)
        for tmp in already_add:
            call_sets[tmp] = call_sets.get(tmp, 0) + 1

    sorted_data = dict(sorted(call_sets.items(), key=lambda item: item[1], reverse=True))
    top_100_set = set()
    for item in sorted_data:
        top_100_set.add(item)
        print(item)
        if len(top_100_set) > 100:
            break
    return top_100_set


def debug_s3():
    top_100_set = get_popular_call()
    # # let's see how many
    theory_files = set()
    total = 0
    for item in os.listdir("extracted_smt"):
        if 'sparse' in item:
            continue

        total += 1
        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)
        temp_call_sets = count_call_types(data)
        remaining_set = temp_call_sets - top_100_set
        if len(remaining_set) == 0:
            theory_files.add(item)
    #
    print(total, len(theory_files))

    # with open("success_files.pickle", "rb") as f:
    #     reality = pickle.load(f)

    # for item in theory_files - reality:
    #     print(item)


if __name__ == "__main__":
    debug_s3()

    # print(reality)
