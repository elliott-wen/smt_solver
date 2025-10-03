import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


def run_file(file):
    """Run main.py with the given file and return (file, returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["python3", "main.py", file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        return file, result.returncode, result.stdout, result.stderr
    except Exception as e:
        return file, -1, "", str(e)


if __name__ == "__main__":
    input_dir = "extracted_smt"
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and "sparse" not in f]

    call_sets = {}
    success_count = 0
    fail_count = 0
    success_set = set()
    # Adjust max_workers if you don't want to use all 256 cores
    with ProcessPoolExecutor(max_workers=16) as executor:


        futures = {executor.submit(run_file, f): f for f in files}

        for future in as_completed(futures):
            file, returncode, stdout, stderr = future.result()
            call_sets[file] = returncode

            if returncode == 0:
                success_count += 1
                success_set.add(file.replace("extracted_smt/", ""))
                os.unlink(file)
                print("OK", file)
            else:
                fail_count += 1
                # Uncomment if you want debugging info:
                # print(f"[FAIL] {file}\nSTDERR:\n{stderr}\n")

    print(f"Total files: {len(call_sets)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    with open("success_files.pickle", "wb") as f:
        pickle.dump(success_set, f)
