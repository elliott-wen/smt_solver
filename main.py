import json
from z3 import EnumSort, IntSort, SeqSort, BoolSort, Int, IntVal, RealVal, BoolVal, BoolRef, ArithRef, Function, Length, \
    If, Xor, Or, And, Not, Solver, sat, Empty, Concat, Unit, ForAll
import re
from json_dump import print_paths
import pickle

# Arg: a tensor handle/id. We use IntSort and keep stable ids per argument index

ArgSort = IntSort()
ArgVector = SeqSort(ArgSort)

# Define a vector as an array Int -> Int
IntVector = SeqSort(IntSort())

DeviceSort, (kCPU) = EnumSort('Device', ['CPU'])

# Tensor property functions
Dim = Function('dim', ArgSort, IntSort())  # rank

ElementSize = Function('element_size', ArgSort, IntSort())

Size = Function('size', ArgSort, IntSort(), IntSort()
                )  # (t, d) -> size[d]
Sizes = Function('size', ArgSort, IntVector
                 )  # t -> sizes

Stride = Function('stride', ArgSort, IntSort(), IntSort()
                  )  # (t,d) -> stride
Strides = Function('strides', ArgSort, IntVector
                   )  # t -> strides

Numel = Function('numel', ArgSort, IntSort())
# (t) -> dtype

DType = Function('dtype', ArgSort, IntSort())

IsContiguous = Function('is_contiguous', ArgSort,
                        BoolSort())  # (t) -> Bool
# (t) -> Device
DeviceOf = Function('device_of', ArgSort, DeviceSort)
LayoutOf = Function('layout_of', ArgSort, IntSort())
IsConj = Function('is_conj', ArgSort,
                  BoolSort())
IsComplex = Function('is_complex', ArgSort, BoolSort())
IsZeroTensor = Function('is_zerotensor', ArgSort, BoolSort())

# Optional extras you might want
RequiresGrad = Function('requires_grad', ArgSort, BoolSort())
IsSparse = Function('is_sparse', ArgSort, BoolSort())
IsNested = Function('is_nested', ArgSort, BoolSort())

# ----------------------------
# Helpers: symbol tables / naming
# ----------------------------


_arg_tensors = {}  # arg index -> IntVal(handle)


def get_arg_tensor(idx: int):
    """Return a stable ArgSort id (IntVal) for the given argument index."""
    if idx not in _arg_tensors:
        _arg_tensors[idx] = Int(f"arg{idx}")
    return _arg_tensors[idx]


def sanitize(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)


# ----------------------------
# ICmp predicate mapping (LLVM CmpInst::Predicate)
# We use Int comparisons for both signed/unsigned here.
# 32 eq, 33 ne, 34 ugt, 35 uge, 36 ult, 37 ule, 38 sgt, 39 sge, 40 slt, 41 sle
# ----------------------------


def icmp_apply(pred: int, lhs, rhs):
    if pred == 32:  # eq
        return lhs == rhs
    if pred == 33:  # ne
        return lhs != rhs
    if pred == 34:  # ugt
        return lhs > rhs
    if pred == 35:  # uge
        return lhs >= rhs
    if pred == 36:  # ult
        return lhs < rhs
    if pred == 37:  # ule
        return lhs <= rhs
    if pred == 38:  # sgt
        return lhs > rhs
    if pred == 39:  # sge
        return lhs >= rhs
    if pred == 40:  # slt
        return lhs < rhs
    if pred == 41:  # sle
        return lhs <= rhs
    raise ValueError(f"Unknown icmp predicate: {pred}")


def call_apply(fn_name, ops):
    if fn_name == "_ZNK3c106Device6is_cpuEv":
        return BoolVal(True)

    # Don't know how to keep track of vector
    if fn_name == "_ZN9__gnu_cxxneIPN2at6TensorESt6vectorIS2_SaIS2_EEEEbRKNS_17__normal_iteratorIT_T0_EESC_":
        return BoolVal(True)

    # Shortcut it
    if fn_name == "_ZNK3c108ArrayRefIlE3vecEv":
        return build_expr(ops[0])

    if fn_name == "_ZNKSt6vectorIlSaIlEE4sizeEv":
        return Length(build_expr(ops[0]))

    if fn_name == "_ZNK3c108ArrayRefIlE4sizeEv":
        return Length(build_expr(ops[0]))

    if fn_name == "_ZNK3c106DeviceeqERKS0_":
        # Shortcut, they should always equal in our fuzzer
        return build_expr(ops[0]) == build_expr(ops[1])

    if fn_name == "_ZN3c10eqEN6caffe28TypeMetaENS_10ScalarTypeE":
        return build_expr(ops[0]) == build_expr(ops[1])

    if fn_name == "_ZN6detail11scalar_typeEN3c1010ScalarTypeE":
        # This function is useless? it just return scalartype anyway?
        return build_expr(ops[0])

    if fn_name == "_ZNK2at10TensorBase5numelEv":
        return Numel(build_expr(ops[0]))

    if fn_name == "_ZN2at6TensorC2ERKS0_":
        return build_expr(ops[0])

    if fn_name == "_ZNR2at6TensoraSERKS0_":
        return build_expr(ops[0])

    if fn_name == "_ZNK2at6Tensor10contiguousEN3c1012MemoryFormatE":
        print("Not implemented yet", fn_name)
        return build_expr(ops[0])

    if fn_name == "_ZNK2at6Tensor4conjEv":
        print("Not implemented yet", fn_name)
        return build_expr(ops[0])

    if fn_name == "_ZNK2at10TensorBase3dimEv":
        return Dim(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase11scalar_typeEv":
        return DType(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase5dtypeEv":
        return DType(build_expr(ops[0]))

    if fn_name == "_ZNK2at18TensorIteratorBase5dtypeEl":
        # If the immediate ops is build, i can handle it so far
        some_sort_array = build_expr(ops[0])
        some_index = build_expr(ops[1])
        return DType(some_sort_array[some_index])

    if fn_name == "_ZN2at20TensorIteratorConfig5buildEv": # Pass through
        return build_expr(ops[0])

    if fn_name == "_ZN2at20TensorIteratorConfig15add_const_inputERKNS_10TensorBaseE":
        return Concat(build_expr(ops[0]), Unit(build_expr(ops[1])))

    if fn_name == "_ZN2at20TensorIteratorConfig10add_outputERKNS_10TensorBaseE":
        return Concat(build_expr(ops[0]), Unit(build_expr(ops[1])))

    if fn_name == "_ZN2at20TensorIteratorConfig20check_all_same_dtypeEb": # Pass through
        return build_expr(ops[0])

    if fn_name == "_ZN2at20TensorIteratorConfig21set_check_mem_overlapEb":
        return build_expr(ops[0])

    if fn_name == "_ZN2at20TensorIteratorConfigC2Ev":
        return Empty(ArgVector)

    if fn_name == "_ZNK2at10TensorBase7is_conjEv":
        return IsConj(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase9is_nestedEv":
        return IsNested(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase10is_complexEv":
        return IsComplex(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase14_is_zerotensorEv":
        return IsZeroTensor(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase6deviceEv":
        return DeviceOf(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase6layoutEv":
        return LayoutOf(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase7stridesEv":
        return Strides(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase6strideEl":
        return Stride(build_expr(ops[0]), build_expr(ops[1]))

    if fn_name == "_ZNK2at10TensorBase5sizesEv":
        return Sizes(build_expr(ops[0]))

    if fn_name == "_ZNK2at10TensorBase4sizeEl":
        return Size(build_expr(ops[0]), build_expr(ops[1]))

    # Handle range here
    if fn_name == "_ZN3c106irangeIiiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_":
        seq_expr = Empty(IntVector)
        for v in range(build_expr(ops[0]).as_long(), build_expr(ops[1]).as_long()):
            seq_expr = Concat(seq_expr, Unit(IntVal(v)))

        return seq_expr

    if fn_name == "_ZN3c106irangeIiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_":
        seq_expr = Empty(IntVector)
        for v in range(build_expr(ops[0]).as_long()):
            seq_expr = Concat(seq_expr, Unit(IntVal(v)))
        return seq_expr

    if fn_name == "_ZNK3c1013integer_rangeIiLb0ELb1EE5beginEv" or fn_name == "_ZNK3c1013integer_rangeIiLb1ELb1EE5beginEv":
        return build_expr(ops[0])[0]  # build_expr(ops[0]) expects to be a seq

    if fn_name == "_ZNK3c106detail16integer_iteratorIiLb0ELi0EEdeEv" or fn_name == "_ZNK3c106detail16integer_iteratorIiLb1ELi0EEdeEv":
        # Degrade to direct access
        return build_expr(ops[0])

    if fn_name == "_ZNK3c108ArrayRefIlEixEm":
        some_sort_array = build_expr(ops[0])
        some_index = build_expr(ops[1])
        return some_sort_array[some_index]

    if fn_name == "_ZNK2at10TensorBase12element_sizeEv":
        sym_input = build_expr(ops[0])
        return ElementSize(sym_input)

    if fn_name == "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_":
        in1 = build_expr(ops[0])
        in2 = build_expr(ops[1])
        return in1 == in2

    raise NotImplementedError(f"Unhandled call: {fn_name}")


# ----------------------------
# Expression builder (from your JSON nodes)
# Each node has shape: {"inst": "...", "ops": [...]} or leaves with "const"
# ----------------------------


def build_expr(node):
    if not isinstance(node, dict):
        raise ValueError(f"Bad node (expected dict): {node}")

    inst = node.get("inst", "").lower()
    ops = node.get("ops", [])

    if inst == "const_int":
        cval = node.get("const")
        return IntVal(int(cval))
    if inst == "const_fp":
        cval = node.get("const")
        # Use Real for FP constants (simplified)
        return RealVal(str(cval))
    if inst == "const_arg":
        cval = node.get("const")
        return get_arg_tensor(int(cval))
    if inst == "const_global":
        cval = node.get("const")
        # Only used as callee inside "call"; standalone -> treat as a symbolic Int
        return Int(sanitize(str(cval)))

    if inst == "load":
        # passthrough the pointee expr
        return build_expr(ops[0])

    if inst == "store":
        # passthrough the pointer expr
        return build_expr(ops[0])

    if inst == "call":
        # ops[0] must be const_global with function name; remaining are arguments
        if not ops or ops[0].get("inst") != "const_global":
            raise ValueError(f"call without const_global callee: {node}")
        fn_name = ops[0]["const"]

        return call_apply(fn_name, ops[1:])

    if inst == "icmp":
        # ops[0]: predicate const_int; ops[1]: lhs; ops[2]: rhs
        if len(ops) != 3 or ops[0].get("inst") != "const_int":
            raise ValueError(f"icmp malformed: {node}")
        pred = int(ops[0]["const"])
        lhs = build_expr(ops[1])
        rhs = build_expr(ops[2])
        return icmp_apply(pred, lhs, rhs)

    if inst == "select":
        # ops: condition, trueVal, falseVal
        cond = build_expr(ops[0])
        tval = build_expr(ops[1])
        fval = build_expr(ops[2])
        return If(cond, tval, fval)

    # Arithmetic / bitwise (simplified to Int arithmetic/bitwise)
    if inst in ("add", "fadd"):
        return build_expr(ops[0]) + build_expr(ops[1])
    if inst in ("sub", "fsub"):
        return build_expr(ops[0]) - build_expr(ops[1])
    if inst in ("mul", "fmul"):
        return build_expr(ops[0]) * build_expr(ops[1])
    if inst in ("sdiv", "udiv", "fdiv", "div"):
        return build_expr(ops[0]) / build_expr(ops[1])
    if inst == "and":
        a, b = build_expr(ops[0]), build_expr(ops[1])
        if isinstance(a, BoolRef) and isinstance(b, BoolRef):
            return And(a, b)
        raise NotImplementedError("bitwise and on int not implemented here.")
    if inst == "or":
        a, b = build_expr(ops[0]), build_expr(ops[1])
        if isinstance(a, BoolRef) and isinstance(b, BoolRef):
            return Or(a, b)
        raise NotImplementedError("bitwise or on int not implemented here.")
    if inst == "xor":
        a, b = build_expr(ops[0]), build_expr(ops[1])
        if isinstance(a, BoolRef) and isinstance(b, BoolRef):
            return Xor(a, b)
        raise NotImplementedError("bitwise xor on int not implemented here.")
    if inst in ("shl", "lshr", "ashr"):
        raise NotImplementedError(
            "shifts not implemented in this int-based model.")

    # Casts (pass-through, since we're using untyped Int/Real/Bools)
    if inst in ("trunc", "sext", "zext", "fptosi", "ptrtoint",
                "sitofp", "inttoptr", "fpext", "uitofp"):
        return build_expr(ops[0])

    raise NotImplementedError(f"Unhandled instruction: {inst}")


# ----------------------------
# Build solvers per path
# ----------------------------


def contains_const_arg(cond):
    """Recursively check if cond or any of its ops contains a const_arg."""
    if not isinstance(cond, dict):
        return False

    # we can shortcut the error
    if cond.get("error", None) is not None:
        raise ValueError("Error detected in op tree")

    if cond.get("inst") == "const_arg":
        return True

    for op in cond.get("ops", []):
        if contains_const_arg(op):
            return True

    return False


def build_solvers_from_data(data):
    solvers = []

    for path_idx, path in enumerate(data.get("paths", []), 1):
        s = Solver()

        path_details = path["detail"]
        for step_idx, step in enumerate(reversed(path_details), 1):
            cond = step.get("condition")
            if cond is None:
                continue

            # Todo we need to shortcut some call
            if not contains_const_arg(cond):
                continue

            is_switch = step.get("isSwitch", False)
            taken = step.get("taken", None)
            excluded = step.get("excludedCases", [])

            if not is_switch:
                # normal branch
                cond_expr = build_expr(cond)
                if isinstance(cond_expr, BoolRef):
                    # print("br:", cond_expr)
                    s.add(cond_expr if taken == 1 else Not(cond_expr))
                elif isinstance(cond_expr, ArithRef):
                    # Interpret nonzero as True
                    if taken:
                        augmented_cond_expr = cond_expr != 0  # taken branch means condition is True
                    else:
                        # not taken branch means condition is False
                        augmented_cond_expr = cond_expr == 0

                    # print("br (int as bool):", augmented_cond_expr)
                    s.add(augmented_cond_expr)
            else:
                # switch handling
                # switch condition is an Int/BitVec
                cond_expr = build_expr(cond)
                # print("switch:", cond_expr)
                if excluded:
                    # Default branch: not equal to any of excludedCases
                    s.add(And([cond_expr != IntVal(v) for v in excluded]))
                else:
                    # Normal case branch
                    s.add(cond_expr == taken)

        solvers.append(s)
    return solvers

# ----------------------------
# Example run (assumes `data` is already loaded with your JSON)
# ----------------------------
if __name__ == "__main__":
    with open(
            "extracted_smt/_ZN2at6native4rollERKNS_6TensorEN3c108ArrayRefIlEES6_.json") as fp:
        data = json.load(fp)
    # with open("extracted_smt/_ZN2at6native4put_ERNS_6TensorERKS1_S4_b.txt") as fp:
    #     data = json.load(fp)
    print_paths(data)

    with open("mapping.pickle", "rb") as f:
        param_mapping = pickle.load(f)

    llvm_name = data['paths'][0]["funcInfo"]['ori_name']
    param_info = param_mapping[llvm_name]

    # Build each fault path independently
    solvers = build_solvers_from_data(data)
    fault_paths = []
    for i, s in enumerate(solvers, 1):
        print(f"\nPath {i} that leads to fault:")
        doom_path_constraints = s.assertions()
        for constraint in doom_path_constraints:
            print(constraint)

        fault_paths.append(And(list(doom_path_constraints)))

    # Not A and Not B
    safe_formula = Not(Or(fault_paths))
    s_neg = Solver()
    s_neg.add(safe_formula)

    # Constraint known function
    x = Int('x')  # dummy variable for ForAll
    s_neg.add(ForAll(x, Or([ElementSize(x) == v for v in [1, 2, 4, 6, 16]])))

    res = s_neg.check()
    if res == sat:
        print("===> Inputs that avoid fault:", s_neg.model())
    else:
        print("No way to avoid fault (path always triggers it).")
