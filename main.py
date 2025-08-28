import json
import re
from param_parser import extract_llvm_param_types
from z3 import *
import logging
from json_dump import print_paths

# ----------------------------
# Tensor symbolic model
# ----------------------------
class TensorModel:
    def __init__(self, argument_map):
        self._arg_tensors = {}
        self.constraints = []  # store all tensor-related constraints

        # Define sorts
        self.ArgSort = IntSort()
        self.ArgVector = SeqSort(self.ArgSort)
        self.IntVector = SeqSort(IntSort())
        self.DeviceSort, (self.kCPU,) = EnumSort("Device", ["CPU"])
        self.TensorSort = DeclareSort("Tensor")

        # Tensor property functions
        self.Dim = Function("dim", self.TensorSort, IntSort())
        self.ElementSize = Function("element_size", self.TensorSort, IntSort())
        self.Size = Function("size", self.TensorSort, IntSort(), IntSort())
        self.Sizes = Function("sizes", self.TensorSort, self.IntVector)
        self.Stride = Function("stride", self.TensorSort, IntSort(), IntSort())
        self.Strides = Function("strides", self.TensorSort, self.IntVector)
        self.Numel = Function("numel", self.TensorSort, IntSort())
        self.DType = Function("dtype", self.TensorSort, IntSort())
        self.IsContiguous = Function("is_contiguous", self.TensorSort, BoolSort())
        self.DeviceOf = Function("device_of", self.TensorSort, self.DeviceSort)
        self.LayoutOf = Function("layout_of", self.TensorSort, IntSort())
        self.IsConj = Function("is_conj", self.TensorSort, BoolSort())
        self.IsComplex = Function("is_complex", self.TensorSort, BoolSort())
        self.IsZeroTensor = Function("is_zerotensor", self.TensorSort, BoolSort())
        self.IsSparse = Function("is_sparse", self.TensorSort, BoolSort())
        self.IsNested = Function("is_nested", self.TensorSort, BoolSort())

        self.argument_map = argument_map

    def get_arg_tensor(self, idx: int):
        if idx not in self._arg_tensors:
            arg_info = self.argument_map[idx]
            if arg_info == "Tensor":
                self._arg_tensors[idx] = Const(f"tensor_arg_{idx}", self.TensorSort)
            elif arg_info == "int":
                self._arg_tensors[idx] = Const(f"int_arg_{idx}", IntSort())
            elif arg_info == "int[]":
                self._arg_tensors[idx] = Const(f"intarray_arg_{idx}", self.IntVector)
            else:
                raise ValueError(f"Unhandled argument type {arg_info}")
        return self._arg_tensors[idx]

    @staticmethod
    def sanitize(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

# ----------------------------
# LLVM ICmp mapper
# ----------------------------
class ICmpMapper:
    @staticmethod
    def apply(pred: int, lhs, rhs):
        if pred == 32: return lhs == rhs
        if pred == 33: return lhs != rhs
        if pred == 34: return lhs > rhs
        if pred == 35: return lhs >= rhs
        if pred == 36: return lhs < rhs
        if pred == 37: return lhs <= rhs
        if pred == 38: return lhs > rhs
        if pred == 39: return lhs >= rhs
        if pred == 40: return lhs < rhs
        if pred == 41: return lhs <= rhs
        raise ValueError(f"Unknown icmp predicate: {pred}")

# ----------------------------
# Function mapping
# ----------------------------
class FunctionMapper:
    def __init__(self, model: TensorModel):
        self.model = model

    # Tensor copy helper: automatically registers constraints
    def make_tensor_copy(self, src_tensor, new_sizes):
        bm = self.model
        new_tensor = Const(f"tensor_copy_{id(src_tensor)}", bm.TensorSort)
        bm.add_constraint(bm.Dim(new_tensor) == bm.Dim(src_tensor))
        bm.add_constraint(bm.Numel(new_tensor) == bm.Numel(src_tensor))
        bm.add_constraint(bm.ElementSize(new_tensor) == bm.ElementSize(src_tensor))
        bm.add_constraint(bm.DeviceOf(new_tensor) == bm.DeviceOf(src_tensor))
        bm.add_constraint(bm.LayoutOf(new_tensor) == bm.LayoutOf(src_tensor))
        bm.add_constraint(bm.IsContiguous(new_tensor) == bm.IsContiguous(src_tensor))
        return new_tensor

    def model_expand_size(self, src_tensor, target_sizes):
        """
        Symbolically model Tensor::expand_size:
        src_tensor: source tensor (Z3 Const)
        target_sizes: Z3 IntVector representing the target shape
        Returns a new Z3 Const representing the expanded tensor
        """
        bm = self.model
        new_tensor = Const(f"tensor_expand_{id(src_tensor)}", bm.TensorSort)

        # Copy basic properties
        bm.add_constraint(bm.ElementSize(new_tensor) == bm.ElementSize(src_tensor))
        bm.add_constraint(bm.DeviceOf(new_tensor) == bm.DeviceOf(src_tensor))
        bm.add_constraint(bm.LayoutOf(new_tensor) == bm.LayoutOf(src_tensor))
        bm.add_constraint(bm.IsContiguous(new_tensor) == bm.IsContiguous(src_tensor))

        # Set expanded shape
        bm.add_constraint(bm.Dim(new_tensor) == Length(target_sizes))
        bm.add_constraint(bm.Sizes(new_tensor) == target_sizes)

        # Optional: Numel as product of sizes (if you want precise numel modeling)
        numel_expr = IntVal(1)
        for i in range(Length(target_sizes)):
            numel_expr *= target_sizes[i]
        bm.add_constraint(bm.Numel(new_tensor) == numel_expr)

        return new_tensor

    # Apply LLVM function call
    def apply(self, fn_name: str, ops):
        bm = self.model

        # ----------------------------
        # Keep all original call_apply cases
        # ----------------------------
        if fn_name == "_ZNK3c1010MaybeOwnedIN2at6TensorEEptEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at6TensorC2Ev":
            logging.warning("Use _ZN2at6TensorC2Ev")
            return Const("tensor_alloca", bm.TensorSort)

        if fn_name == "_ZN2at6native17use_mkldnn_matmulERKNS_6TensorES3_S3_":
            return BoolVal(False)

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
            logging.warning("Not implemented yet: %s", fn_name)
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

        # Tensor clone/detach modeled as copy
        if fn_name == "_ZN2at11expand_sizeERKNS_6TensorEN3c108ArrayRefIlEEPKc":
            return self.model_expand_size(self.build_expr(ops[0]), self.build_expr(ops[1]))

        if fn_name == "_ZN3c108ArrayRefIlEC2ERKSt16initializer_listIlE":
            # Passthrough
            return self.build_expr(ops[0])

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

    # Recursive expression builder
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

    def build_solver_from_data(self, path, solver):
        for step in reversed(path.get("detail", [])):
            cond = step.get("condition")
            if cond is None or not self.contains_const_arg(cond):
                continue
            expr = self.mapper.build_expr(cond)
            if isinstance(expr, BoolRef):
                solver.add(expr if step.get("taken") else Not(expr))
            elif isinstance(expr, ArithRef):
                taken = step.get("taken")
                solver.add(expr != 0 if taken else expr == 0)

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

    solvers = []
    solver_builder = SolverBuilder(model, mapper)
    for path in data.get("paths", []):
        solver = Solver()
        solver_builder.build_solver_from_data(path, solver)
        print(solver.assertions())
        solvers.append(solver)

    fault_paths = [And(s.assertions()) for s in solvers]
    safe_formula = Not(Or(fault_paths))
    s_neg = Solver()
    s_neg.add(safe_formula)

    # add model-wide constraints if any
    for c in model.constraints:
        s_neg.add(c)

    res = s_neg.check()
    if res == sat:
        print("Inputs that avoid fault:", s_neg.model())
    else:
        print("No way to avoid fault.")

if __name__ == "__main__":
    main("extracted_smt/_ZN2at6native4addrERKNS_6TensorES3_S3_RKN3c106ScalarES7_.json")
