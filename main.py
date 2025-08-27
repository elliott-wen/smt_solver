import json
import re
from param_parser import extract_llvm_param_types
from z3 import *

from json_dump import print_paths


# ----------------------------
# Tensor symbolic model
# ----------------------------
class TensorModel:
    def __init__(self, argument_map):
        self._arg_tensors = {}

        # Define sorts
        self.ArgSort = IntSort()
        self.ArgVector = SeqSort(self.ArgSort)
        self.IntVector = SeqSort(IntSort())
        self.DeviceSort, (self.kCPU,) = EnumSort("Device", ["CPU"])
        self.TensorSort = DeclareSort("Tensor")

        # Tensor property functions
        self.Dim = Function("dim", self.ArgSort, IntSort())
        self.ElementSize = Function("element_size", self.ArgSort, IntSort())
        self.Size = Function("size", self.ArgSort, IntSort(), IntSort())
        self.Sizes = Function("sizes", self.ArgSort, self.IntVector)
        self.Stride = Function("stride", self.ArgSort, IntSort(), IntSort())
        self.Strides = Function("strides", self.ArgSort, self.IntVector)
        self.Numel = Function("numel", self.ArgSort, IntSort())
        self.DType = Function("dtype", self.ArgSort, IntSort())
        self.IsContiguous = Function("is_contiguous", self.ArgSort, BoolSort())
        self.DeviceOf = Function("device_of", self.ArgSort, self.DeviceSort)
        self.LayoutOf = Function("layout_of", self.ArgSort, IntSort())
        self.IsConj = Function("is_conj", self.ArgSort, BoolSort())
        self.IsComplex = Function("is_complex", self.ArgSort, BoolSort())
        self.IsZeroTensor = Function("is_zerotensor", self.ArgSort, BoolSort())
        self.RequiresGrad = Function("requires_grad", self.ArgSort, BoolSort())
        self.IsSparse = Function("is_sparse", self.ArgSort, BoolSort())
        self.IsNested = Function("is_nested", self.ArgSort, BoolSort())

        self.argument_map = argument_map

    def get_arg_tensor(self, idx: int):
        if idx not in self._arg_tensors:
            arg_info = self.argument_map[idx]
            if arg_info == "Tensor":
                self._arg_tensors[idx] = Const(f"tensor_arg_{idx}", self.TensorSort)
            elif arg_info == "int":
                self._arg_tensors[idx] = Const(f"tensor_int_{idx}", IntSort)
            elif arg_info == "int[]":
                self._arg_tensors[idx] = Const(f"tensor_intarray_{idx}", self.IntVector)
            else:
                raise ValueError(f"Unhandled argument type {arg_info}")
        return self._arg_tensors[idx]

    @staticmethod
    def sanitize(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ----------------------------
# LLVM ICmp mapper
# ----------------------------
class ICmpMapper:
    @staticmethod
    def apply(pred: int, lhs, rhs):
        mapping = {
            32: lhs == rhs, 33: lhs != rhs,
            34: lhs > rhs, 35: lhs >= rhs,
            36: lhs < rhs, 37: lhs <= rhs,
            38: lhs > rhs, 39: lhs >= rhs,
            40: lhs < rhs, 41: lhs <= rhs,
        }
        if pred not in mapping:
            raise ValueError(f"Unknown icmp predicate: {pred}")
        return mapping[pred]


# ----------------------------
# Function mapping (merged call_apply)
# ----------------------------
class FunctionMapper:
    def __init__(self, model: TensorModel):
        self.model = model

    def apply(self, fn_name: str, ops):
        bm = self.model

        # ---- Direct paste of your call_apply cases ----

        if fn_name == "_ZNK3c106Device6is_cpuEv":
            return BoolVal(True)

        if fn_name == "_ZN9__gnu_cxxneIPN2at6TensorESt6vectorIS2_SaIS2_EEEEbRKNS_17__normal_iteratorIT_T0_EESC_":
            return BoolVal(True)

        if fn_name == "_ZNK3c108ArrayRefIlE3vecEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK3c108ArrayRefIlE5emptyEv":
            return Length(self.build_expr(ops[0])) == 0

        if fn_name in ("_ZNKSt6vectorIlSaIlEE4sizeEv", "_ZNK3c108ArrayRefIlE4sizeEv"):
            return Length(self.build_expr(ops[0]))

        if fn_name == "_ZNK3c106DeviceeqERKS0_":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZN3c10eqEN6caffe28TypeMetaENS_10ScalarTypeE":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZN6detail11scalar_typeEN3c1010ScalarTypeE":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase5numelEv":
            return bm.Numel(self.build_expr(ops[0]))

        if fn_name in ("_ZN2at6TensorC2ERKS0_", "_ZNR2at6TensoraSERKS0_"):
            return self.build_expr(ops[0])

        if fn_name in ("_ZNK2at6Tensor10contiguousEN3c1012MemoryFormatE",
                       "_ZNK2at6Tensor4conjEv"):
            print("Not implemented yet", fn_name)
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase3dimEv":
            return bm.Dim(self.build_expr(ops[0]))

        if fn_name in ("_ZNK2at10TensorBase11scalar_typeEv", "_ZNK2at10TensorBase5dtypeEv"):
            return bm.DType(self.build_expr(ops[0]))

        if fn_name == "_ZNK2at18TensorIteratorBase5dtypeEl":
            arr, idx = self.build_expr(ops[0]), self.build_expr(ops[1])
            return bm.DType(arr[idx])

        if fn_name == "_ZN2at20TensorIteratorConfig5buildEv":
            return self.build_expr(ops[0])

        if fn_name in ("_ZN2at20TensorIteratorConfig15add_const_inputERKNS_10TensorBaseE",
                       "_ZN2at20TensorIteratorConfig10add_outputERKNS_10TensorBaseE"):
            return Concat(self.build_expr(ops[0]), Unit(self.build_expr(ops[1])))

        if fn_name in ("_ZN2at20TensorIteratorConfig20check_all_same_dtypeEb",
                       "_ZN2at20TensorIteratorConfig21set_check_mem_overlapEb"):
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at20TensorIteratorConfigC2Ev":
            return Empty(bm.ArgVector)

        if fn_name == "_ZNK2at10TensorBase7is_conjEv":
            return bm.IsConj(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase9is_nestedEv":
            return bm.IsNested(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase10is_complexEv":
            return bm.IsComplex(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase14_is_zerotensorEv":
            return bm.IsZeroTensor(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase6deviceEv":
            return bm.DeviceOf(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase6layoutEv":
            return bm.LayoutOf(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase7stridesEv":
            return bm.Strides(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase6strideEl":
            return bm.Stride(self.build_expr(ops[0]), self.build_expr(ops[1]))
        if fn_name == "_ZNK2at10TensorBase5sizesEv":
            return bm.Sizes(self.build_expr(ops[0]))
        if fn_name == "_ZNK2at10TensorBase4sizeEl":
            return bm.Size(self.build_expr(ops[0]), self.build_expr(ops[1]))

        # irange handling
        if fn_name.startswith("_ZN3c106irange"):
            start = self.build_expr(ops[0]).as_long()
            end = self.build_expr(ops[1]).as_long() if len(ops) > 1 else start
            seq_expr = Empty(bm.IntVector)
            for v in range(start, end):
                seq_expr = Concat(seq_expr, Unit(IntVal(v)))
            return seq_expr

        if fn_name.startswith("_ZNK3c1013integer_range") and "beginEv" in fn_name:
            return self.build_expr(ops[0])[0]

        if fn_name.startswith("_ZNK3c106detail16integer_iterator") and "deEv" in fn_name:
            return self.build_expr(ops[0])

        if fn_name == "_ZNK3c108ArrayRefIlEixEm":
            arr, idx = self.build_expr(ops[0]), self.build_expr(ops[1])
            return arr[idx]

        if fn_name == "_ZNK2at10TensorBase12element_sizeEv":
            return bm.ElementSize(self.build_expr(ops[0]))

        if fn_name == "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        raise NotImplementedError(f"Unhandled call: {fn_name}")

    # ----------------------------
    # Expression builder
    # ----------------------------
    def build_expr(self, node):
        inst = node.get("inst")
        ops = node.get("ops", [])

        if inst == "const_int":
            return IntVal(int(node["const"]))
        if inst == "const_arg":
            return self.model.get_arg_tensor(int(node["const"]))
        if inst == "const_global":
            return Int(self.model.sanitize(str(node["const"])))
        if inst in ("load", "store"):
            return self.build_expr(ops[0])
        if inst == "call":
            fn_name = ops[0]["const"]
            return self.apply(fn_name, ops[1:])
        if inst == "icmp":
            pred = int(ops[0]["const"])
            lhs, rhs = self.build_expr(ops[1]), self.build_expr(ops[2])
            return ICmpMapper.apply(pred, lhs, rhs)
        if inst == "select":
            cond, tval, fval = map(self.build_expr, ops)
            return If(cond, tval, fval)
        if inst in ("add", "fadd"):
            return self.build_expr(ops[0]) + self.build_expr(ops[1])
        if inst in ("sub", "fsub"):
            return self.build_expr(ops[0]) - self.build_expr(ops[1])
        if inst in ("mul", "fmul"):
            return self.build_expr(ops[0]) * self.build_expr(ops[1])
        if inst in ("sdiv", "udiv", "fdiv", "div"):
            return self.build_expr(ops[0]) / self.build_expr(ops[1])
        if inst in ("trunc", "sext", "zext", "fptosi", "ptrtoint",
                    "sitofp", "inttoptr", "fpext", "uitofp"):
            return self.build_expr(ops[0])

        raise NotImplementedError(f"Unhandled instruction: {inst}")


# ----------------------------
# Solver builder
# ----------------------------
class SolverBuilder:
    def __init__(self, model: TensorModel, mapper: FunctionMapper):
        self.model = model
        self.mapper = mapper

    def contains_const_arg(self, cond):
        if not isinstance(cond, dict):
            return False
        if cond.get("inst") == "const_arg":
            return True
        return any(self.contains_const_arg(op) for op in cond.get("ops", []))

    def build_solvers_from_data(self, data):
        solvers = []
        for path in data.get("paths", []):
            s = Solver()
            for step in reversed(path.get("detail", [])):
                cond = step.get("condition")
                if cond is None or not self.contains_const_arg(cond):
                    continue
                expr = self.mapper.build_expr(cond)
                if isinstance(expr, BoolRef):
                    s.add(expr if step.get("taken") else Not(expr))
                elif isinstance(expr, ArithRef):
                    taken = step.get("taken")
                    s.add(expr != 0 if taken else expr == 0)
            solvers.append(s)
        return solvers


# ----------------------------
# Main
# ----------------------------
def main(input_file):
    with open(input_file) as fp:
        data = json.load(fp)
    print_paths(data)

    llvm_name = data['paths'][0]['funcInfo']['llvm_name']
    llvm_params = data['paths'][0]['funcInfo']['llvm_params']
    argument_map = extract_llvm_param_types(llvm_name)
    if "sret_" in llvm_params[0]:
        argument_map.insert(0, "Tensor")

    model = TensorModel(argument_map)
    mapper = FunctionMapper(model)

    solver_builder = SolverBuilder(model, mapper)
    solvers = solver_builder.build_solvers_from_data(data)
    fault_paths = [And(s.assertions()) for s in solvers]

    safe_formula = Not(Or(fault_paths))
    s_neg = Solver()
    s_neg.add(safe_formula)

    # Example domain constraints
    x = Int("x")
    s_neg.add(ForAll(x, Or([model.ElementSize(x) == v for v in [1, 2, 4, 6, 16]])))

    res = s_neg.check()
    if res == sat:
        print("Inputs that avoid fault:", s_neg.model())
    else:
        print("No way to avoid fault.")


if __name__ == "__main__":
    main("extracted_smt/_ZN2at6native4rollERKNS_6TensorEN3c108ArrayRefIlEES6_.json")