import json
import os

torch_check_set = set()
for item in os.listdir("extracted_smt"):
    if item.endswith(".txt"):
        file = os.path.join("extracted_smt", item)
        print(file)
        with open(file) as fp:
            data = json.load(fp)

        for path in data["paths"]:
            if "info" in path:
                info = path["info"]
                if "line" in info:
                    path_id = info["file"] + str(info["line"])

                    torch_check_set.add(path_id)

print(len(torch_check_set))
