import yaml
import re
import os
import json


# import pickle


# def split_top_level_commas(s: str):
#     """Split string by commas at top-level, respecting () and [] nesting."""
#     parts, buf, depth = [], [], 0
#     for ch in s:
#         if ch in "([":
#             depth += 1
#         elif ch in "])":
#             depth -= 1
#         if ch == "," and depth == 0:
#             parts.append("".join(buf).strip())
#             buf = []
#         else:
#             buf.append(ch)
#     if buf:
#         parts.append("".join(buf).strip())
#     return parts
#
#
# def clean_type(type_str: str):
#     """Remove alias annotations like (a), (a!), (a -> *)."""
#     return re.sub(r"\([^)]*\)", "", type_str).strip()
#
#
# def parse_param(param_str: str):
#     """Parse a single parameter like 'Tensor self=default'."""
#     if param_str.strip() in {"*", "/"}:
#         return None  # keyword-only (*) or positional-only (/) marker
#
#     if "=" in param_str:
#         type_and_name, default_val = param_str.split("=", 1)
#         default_val = default_val.strip()
#     else:
#         type_and_name, default_val = param_str, None
#
#     parts = type_and_name.strip().split()
#     if len(parts) < 2:
#         raise ValueError(f"Could not parse param: {param_str}")
#
#     type_str, name = clean_type(parts[0]), parts[1]
#     return {
#         "type": type_str,
#         "name": name,
#         "default": default_val,
#     }
#
#
# def parse_return_type(ret_str: str):
#     """Parse return type(s), ignoring alias annotations."""
#     ret_str = ret_str.strip()
#     if ret_str == "()":
#         return []
#     if ret_str.startswith("(") and ret_str.endswith(")"):
#         inner = ret_str[1:-1].strip()
#         return [clean_type(t) for t in split_top_level_commas(inner)]
#     return [clean_type(ret_str)]
#
#
# def parse_func_signature(signature: str):
#     """Parse a native_functions.yaml func signature into dict."""
#     match = re.match(r"^([A-Za-z0-9_.]+)\((.*)\)\s*->\s*(.*)$", signature.strip())
#     if not match:
#         raise ValueError(f"Invalid function signature: {signature}")
#     func_name, args_str, ret_str = match.groups()
#
#     params = []
#     if args_str.strip():
#         for arg in split_top_level_commas(args_str):
#             param = parse_param(arg)
#             if param:
#                 params.append(param)
#
#     return {
#         "name": func_name,
#         "params": params,
#         "return_type": parse_return_type(ret_str),
#     }
#
#
# def load_func_db():
#     yaml_file = "native_functions.yaml"
#
#     with open(yaml_file, 'r') as f:
#         data = yaml.safe_load(f) or []
#
#     cpu_functions = {}
#
#     for item in data:
#         # Only process functions with dispatch section
#         if not isinstance(item, dict) or 'dispatch' not in item:
#             continue
#
#         original_func = item.get('func')
#         if not original_func:
#             continue
#
#         parse_result = parse_func_signature(original_func)
#         dispatch = item['dispatch']
#
#         for key, func_name in dispatch.items():
#             # normalize subkeys (split by comma, strip spaces)
#             subkeys = [k.strip() for k in key.split(",")]
#
#             for subkey in subkeys:
#                 if subkey.endswith("CPU"):
#                     if func_name not in cpu_functions:
#                         cpu_functions[func_name] = []
#                     # only append if not already present
#                     if parse_result not in cpu_functions[func_name]:
#                         cpu_functions[func_name].append(parse_result)
#
#     return cpu_functions
#

def clean_signature(sig: str) -> str:
    """
    Remove trailing (.0.0.0) or similar suffixes from LLVM symbol strings.
    """
    # remove trailing " (.digits.digits.digits)" part
    return re.sub(r'\s*\(\.\d+(\.\d+)*\)\s*$', '', sig)


def split_params(param_str: str):
    """Split C++ parameter list by commas, ignoring template commas."""
    params, depth, current = [], 0, []
    for ch in param_str:
        if ch == '<':
            depth += 1
            current.append(ch)
        elif ch == '>':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        params.append("".join(current).strip())
    return params


def normalize_llvm_type(ty: str) -> str:
    ty = ty.replace("const", "").replace("&", "").replace("*", "").strip()

    # ---- Tensor-like ----
    if ty.startswith("at::Tensor"):
        return "Tensor"
    if ty.startswith("std::optional<at::Tensor") or ty.startswith("c10::optional<at::Tensor"):
        return "Tensor?"

    # ---- Array of Tensors ----
    if re.match(r"c10::ArrayRef<\s*at::Tensor\s*>", ty):
        return "Tensor[]"
    if re.match(r"std::optional<c10::ArrayRef<\s*at::Tensor\s*>>", ty):
        return "Tensor[]?"

    # ---- Scalar ----
    if "Scalar" in ty and "Tensor" not in ty:
        return "Scalar"

    # ---- Primitive scalars ----
    if ty == "bool":
        return "bool"
    if ty in ("int", "int64_t", "long"):
        return "int"
    if ty in ("double", "float"):
        return "float"

    # ---- Array / shape types ----
    if "IntArrayRef" in ty or "c10::ArrayRef<long>" in ty:
        return "int[]"
    if "SymIntArrayRef" in ty:
        return "int[]"
    # if "c10::SymInt" in ty:
    #     return "SymInt"
    if "c10::OptionalArrayRef<long>" in ty:
        return "int[]?"  # nullable int list
    if "DimnameList" in ty:
        return "int[]"
    if ty.startswith("c10::SmallVector<long"):
        return "int[]"

    # ---- Optional primitives ----
    if ty.startswith("c10::optional<int") or ty.startswith("std::optional<int"):
        return "int?"
    if ty.startswith("c10::optional<double") or ty.startswith("std::optional<double"):
        return "float?"
    if ty.startswith("c10::optional<bool") or ty.startswith("std::optional<bool"):
        return "bool?"
    if ty.startswith("std::optional<c10::Layout"):
        return "int?"
    if ty.startswith("std::optional<long"):
        return "int?"
    if ty.startswith("std::optional<c10::MemoryFormat"):
        return "int?"
    if ty.startswith("std::optional<c10::Device"):
        return "str?"

    if ty.startswith("std::basic_string_view<char,"):
        return "str"

    if "std::optional<std::basic_string_view" in ty or "c10::optional<c10::string_view" in ty:
        return "str?"

    # ---- Generators ----
    # Optional generator
    if ty.startswith("std::optional<at::Generator") or ty.startswith("c10::optional<at::Generator"):
        return "Generator?"

    # ---- Memory / storage ----
    # if "at::Storage" in ty:
    #     return "Storage"
    # if "at::Layout" in ty:
    #     return "Layout"
    # if "at::Device" in ty:
    #     return "Device"
    # if "at::ScalarType" in ty:
    #     return "ScalarType"

    # ---- Default fallback ----
    # return ty
    raise ValueError(f"Unknown type {ty}")


def extract_llvm_param_types(signature: str):
    """
    Extract normalized parameter types from an LLVM signature string.
    """
    sig = clean_signature(signature)
    m = re.search(r'\((.*)\)', sig)
    if not m:
        return []
    raw_params = split_params(m.group(1))
    return [normalize_llvm_type(p) for p in raw_params]


if __name__ == "__main__":

    llvm_name_map = {}

    for item in os.listdir("extracted_smt"):
        file = os.path.join("extracted_smt", item)
        with open(file, 'r') as f:
            data = json.load(f)
        torch_name = data['paths'][0]["funcInfo"]['torch_name']
        llvm_name = data['paths'][0]["funcInfo"]['llvm_name']
        llvm_params = data['paths'][0]["funcInfo"]['llvm_params']

        llvm_types = extract_llvm_param_types(llvm_name)
