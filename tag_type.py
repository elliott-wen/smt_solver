import json
import os


def debug_s3():
    call_set = set()
    for item in os.listdir("extracted_smt"):
        if 'sparse' in item:
            continue

        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)

        for path in data.get("paths", []):
            for path in path['detail']:
                if 'condition' in path:
                    condition = path['condition']
                    if condition['inst'] == 'call':
                        call_name = condition['ops'][0]['const']
                        call_set.add(call_name)

                    elif condition['inst'] in ('icmp', 'fcmp', 'select', 'trunc', 'xor', 'load', 'and', 'or'):
                        print(condition['ops'])
                    elif condition['inst'] == 'const_arg':
                        pass
                    else:
                        print(condition, file)
                        raise NotImplementedError(condition['inst'])

    print(len(call_set))
    for item in call_set:
        print(item)


if __name__ == "__main__":
    debug_s3()
