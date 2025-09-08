import json
import re
import uuid

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
        self.IntArray = SeqSort(IntSort())
        self.TensorStruct = Datatype('Tensor')
        self.TensorStruct.declare(
            'tensor_structure',
            ('sizes', self.IntArray),
            ('dtype', IntSort()),
        )
        self.TensorStruct = self.TensorStruct.create()

        self.ScalarStruct = Datatype('Scalar')
        self.ScalarStruct.declare(
            'scalar_structure',
            ('dtype', IntSort()),
        )
        self.ScalarStruct = self.ScalarStruct.create()

        self.argument_map = argument_map

        self.checked_contiguous = Bool("checked_contiguous")
        self.checked_tensor_name = Bool("checked_tensor_name")
        self.checked_tensor_conj = Bool("checked_tensor_conj")

    def get_arg_tensor(self, idx: int):
        if idx not in self._arg_tensors:
            arg_info = self.argument_map[idx]
            if arg_info == "Tensor":
                self._arg_tensors[idx] = Const(f"tensor_arg_{idx}", self.TensorStruct)
            elif arg_info == "int":
                self._arg_tensors[idx] = Const(f"int_arg_{idx}", IntSort())
            elif arg_info == "int[]":
                self._arg_tensors[idx] = Const(f"intarray_arg_{idx}", self.IntArray)
            elif arg_info == "float":
                self._arg_tensors[idx] = Const(f"float_arg_{idx}", RealSort())
            elif arg_info == "Scalar":
                self._arg_tensors[idx] = Const(f"scalar_arg_{idx}", self.ScalarStruct)
            elif arg_info == "float?":
                self._arg_tensors[idx] = Const(f"optional_float_arg_{idx}", SeqSort(RealSort()))
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
        print(lhs, rhs)
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


class FCmpMapper:
    @staticmethod
    def apply(pred: int, lhs: float, rhs: float):
        if pred == 0:  # False
            return False
        if pred == 1:  # OEQ
            return lhs == rhs
        if pred == 2:  # OGT
            return lhs > rhs
        if pred == 3:  # OGE
            return lhs >= rhs
        if pred == 4:  # OLT
            return lhs < rhs
        if pred == 5:  # OLE
            return lhs <= rhs
        if pred == 6:  # ONE
            return lhs != rhs
        if pred == 9:  # UEQ
            return lhs == rhs
        if pred == 10:  # UGT
            return lhs > rhs
        if pred == 11:  # UGE
            return lhs >= rhs
        if pred == 12:  # ULT
            return lhs < rhs
        if pred == 13:  # ULE
            return lhs <= rhs
        if pred == 14:  # UNE
            return lhs != rhs
        if pred == 15:  # True
            return True

        raise ValueError(f"Unknown fcmp predicate: {pred}")


# ----------------------------
# Helper function
# ----------------------------

def update_tensor_options_dtype(bm, original, updated_scalar_type):
    """
    Return a new TensorOptions with the dtype updated,
    keeping other fields the same.
    """
    return bm.TensorStruct.tensor_structure(bm.TensorStruct.sizes(original), updated_scalar_type)


def update_tensor_options_sizes(bm, original, updated_size):
    """
    Return a new TensorOptions with the dtype updated,
    keeping other fields the same.
    """
    print("original", original)
    print("updated_size", updated_size)
    return bm.TensorStruct.tensor_structure(updated_size, bm.TensorStruct.dtype(original))


def copy_tensor(bm, original):
    return bm.TensorStruct.tensor_structure(bm.TensorStruct.sizes(original), bm.TensorStruct.dtype(original))


# ----------------------------
# Function mapping
# ----------------------------
class FunctionMapper:
    def __init__(self, model: TensorModel):
        self.model = model
        self.new_tensor_id = 0

    # Apply LLVM function call
    def apply(self, fn_name: str, ops):
        bm = self.model

        if fn_name == "_ZN6detail11scalar_typeEN3c1010ScalarTypeE":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase11scalar_typeEv":
            return bm.TensorStruct.dtype(self.build_expr(ops[0]))

        if fn_name == "_ZNK2at18TensorIteratorBase5dtypeEl":
            arr, idx = self.build_expr(ops[0]), self.build_expr(ops[1])
            return bm.TensorStruct.dtype(arr[idx])

        if fn_name == "_ZNK2at10TensorBase5sizesEv":
            return bm.TensorStruct.sizes(self.build_expr(ops[0]))

        if fn_name == "llvm.memcpy.p0.p0.i64":
            return self.build_expr(ops[0])

        if fn_name == "_ZNSt8optionalIN3c1012MemoryFormatEEC2ESt9nullopt_t":
            # Assume it is contigous
            seq_expr = Empty(SeqSort(IntSort))
            return seq_expr

        if fn_name == "_ZN3c1013TensorOptionsC2Ev":
            return Const(f'option_{uuid.uuid4().hex}', bm.TensorStruct)

        if fn_name == "_ZNK2at10TensorBase6deviceEv":
            return String('cpu')

        if fn_name == "_ZNK2at10TensorBase13is_contiguousEN3c1012MemoryFormatE":
            # We can mark this function check contiguous
            bm.add_constraint(bm.checked_contiguous == True)
            return BoolVal(True)

        if fn_name == "_ZNK3c1013TensorOptions5dtypeESt8optionalINS_10ScalarTypeEE":
            # We should just return a tensor option with
            original_tensor_option = self.build_expr(ops[0])
            updated_scalar_type = self.build_expr(ops[1])
            return update_tensor_options_dtype(bm, original_tensor_option, updated_scalar_type)

        if fn_name == "_ZN2at5emptyEN3c108ArrayRefIlEENS0_13TensorOptionsESt8optionalINS0_12MemoryFormatEE":
            # Create a thing, at::empty(c10::ArrayRef<long>, c10::TensorOptions, std::optional<c10::MemoryFormat>)
            updated_size = self.build_expr(ops[0])
            original_tensor_options = self.build_expr(ops[1])
            return update_tensor_options_sizes(bm, original_tensor_options, updated_size)

        if fn_name == "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZNKSt8optionalIN3c1012MemoryFormatEE9has_valueEv":
            return Length(self.build_expr(ops[0])) > 0

        if fn_name == "_ZN2at13globalContextEv":
            raise NotImplementedError("UnImplemented _ZN2at13globalContextEv")

        if fn_name == "_ZNK2at10TensorBase10is_complexEv":
            stype = bm.TensorStruct.dtype(self.build_expr(ops[0]))
            return And(stype != 8, stype != 9, stype != 10)

        if fn_name == "_ZNK2at7Context23deterministicAlgorithmsEv":
            return BoolVal(False)

        if fn_name == "_ZNK2at10TensorBase19unsafeGetTensorImplEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase17is_floating_pointEv":
            stype = bm.ScalarStruct.dtype(self.build_expr(ops[0]))
            # Floating types: Half(5), Float(6), Double(7), BFloat16(15)
            is_floating = Or(stype == 5, stype == 6, stype == 7, stype == 15)
            return is_floating

        if fn_name == "_ZNR3c1016OptionalArrayRefIlE5valueEv":
            # return the first element in the list
            optional_structure = self.build_expr(ops[0])
            return optional_structure[0]

        if fn_name == "_ZNK2at18TensorIteratorBase12common_dtypeEv":
            # Let's use the first input
            tensor_array = self.build_expr(ops[0])
            return bm.TensorStruct.dtype(tensor_array[1])

        if fn_name == "_ZN3c108ArrayRefIlEC2Ev":
            seq_expr = Empty(SeqSort(IntSort))
            return seq_expr

        if fn_name == "_ZNK2at10TensorBase6is_cpuEv":
            raise BoolVal(True)

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2ESt9nullopt_t":
            seq_expr = Empty(SeqSort(SeqSort(IntSort)))  # The first seq is for optional, the second is for array
            return seq_expr

        if fn_name == "_ZNK2at10TensorBase5dtypeEv":
            return bm.TensorStruct.dtype(self.build_expr(ops[0]))

        if fn_name == "_ZNK3c1016OptionalArrayRefIlEcvbEv":
            return Length(self.build_expr(ops[0])) > 0

        if fn_name == "_ZNK3c107StoragecvbEv":
            return BoolVal(True)

        if fn_name == "_ZNK3c1010TensorImpl14unsafe_storageEv":
            raise NotImplementedError("_ZNK3c1010TensorImpl14unsafe_storageEv")

        if fn_name == "_ZN3c1010TensorImpl13generic_sizesIlEENS_8ArrayRefIT_EEv":
            return bm.TensorStruct.sizes(self.build_expr(ops[0]))

        if fn_name == "_ZN3c1010TensorImpl15generic_stridesIlEENS_8ArrayRefIT_EEv":
            # Wrong implementation
            return bm.TensorStruct.sizes(self.build_expr(ops[0]))

        if fn_name == "_ZNK3c1011StorageImpl9resizableEv":
            # We assume it is resizble
            return BoolVal(True)

        if fn_name == "_ZNK3c107Storage20unsafeGetStorageImplEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase9has_namesEv":
            bm.add_constraint(bm.checked_tensor_name == True)
            return BoolVal(False)

        if fn_name == "_ZNK3c1013TensorOptions6deviceIJNS_6DeviceEEEES0_DpOT_":
            raise self.build_expr(ops[0])

        if fn_name == "_ZNSt8optionalIN3c1010ScalarTypeEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            # Take scalartype and construct optional
            seq_expr = Empty(SeqSort(IntSort))
            seq_expr = Concat(seq_expr, Unit(self.build_expr(ops[0])))
            return seq_expr

        if fn_name == "makeInitlaizerList":
            content = self.build_expr(ops[0])
            if is_seq(content):
                # Good enough
                return content
            else:
                # It may be a store inside
                seq_expr = Empty(bm.IntArray)
                seq_expr = Concat(seq_expr, Unit(content))
                return seq_expr

        if fn_name == "_ZN2at6native20_resize_output_checkIlEEbRKNS_6TensorEN3c108ArrayRefIT_EE":
            logging.warn("_ZN2at6native20_resize_output_checkIlEEbRKNS_6TensorEN3c108ArrayRefIT_EE is called")
            return BoolVal(True)

        if fn_name == "_ZN2at20isTensorSubclassLikeERKNS_6TensorE":
            # We may just set it true?
            logging.warn("_ZN2at20isTensorSubclassLikeERKNS_6TensorE is called")
            return BoolVal(True)

        if fn_name == "_ZNK2at6Tensor10contiguousEN3c1012MemoryFormatE":
            bm.add_constraint(bm.checked_contiguous == True)
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase9is_nestedEv":
            # Let's not do nested?
            logging.warn("_ZNK2at10TensorBase9is_nestedEv is called, return False for now")
            return BoolVal(False)

        if fn_name == "_ZNK3c106detail16integer_iteratorIlLb1ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorIlLb1ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeIlLb1ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIlLb1ELb1EE5beginEv")

        if fn_name == "_ZN3c106irangeIlTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIlTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_")

        if fn_name == "_ZNK3c1013integer_rangeIlLb1ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIlLb1ELb1EE3endEv")

        if fn_name == "_ZN3c108ArrayRefIlEC2ERKSt16initializer_listIlE":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase7is_sameERKS0_":
            raise self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZNK2at10TensorBase7definedEv":
            # We will just pump defined tensor
            return BoolVal(True)

        if fn_name == "_ZNK2at10TensorBase7optionsEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase6layoutEv":
            # Always dense
            return IntVal(0)

        if fn_name == "_ZNR2at6TensoraSEOS0_":
            raise NotImplementedError(f"Unhandled call: _ZNR2at6TensoraSEOS0_")

        if fn_name == "_ZN2at20TensorIteratorConfig5buildEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at20TensorIteratorConfigC2Ev":
            return Empty(SeqSort(bm.TensorStruct))

        if fn_name == "_ZNK2at10TensorBase9is_sparseEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase9is_sparseEv")

        if fn_name == "_ZN3c1021isReducedFloatingTypeENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1021isReducedFloatingTypeENS_10ScalarTypeE")

        if fn_name == "_ZN2at6TensorC2Ev":
            return Const(f'empty_tensor_{uuid.uuid4().hex}', bm.TensorStruct)

        if fn_name == "_ZN2at6TensorC2ERKS0_":
            return copy_tensor(bm, self.build_expr(ops[0]))

        if fn_name == "_ZNK2at6Tensor12is_coalescedEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12is_coalescedEv")

        if fn_name == "_ZNKR3c1010MaybeOwnedIN2at6TensorEEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNKR3c1010MaybeOwnedIN2at6TensorEEdeEv")

        if fn_name == "_ZNK3c106DeviceeqERKS0_":
            return BoolVal(True)

        if fn_name == "_ZNK2at10TensorBase3dimEv":
            return Length(bm.TensorStruct.sizes(self.build_expr(ops[0])))

        if fn_name == "_ZNK2at10TensorBase4sizeEl":
            return bm.TensorStruct.sizes(self.build_expr(ops[0]))

        if fn_name == "_ZN6caffe2eqERKNS_8TypeMetaES2_":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe2eqERKNS_8TypeMetaES2_")

        if fn_name == "_ZNKSt8optionalIlE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIlE9has_valueEv")

        if fn_name == "_ZN2at20TensorIteratorConfig10add_outputERKNS_10TensorBaseE":
            return Concat(self.build_expr(ops[0]), Unit(self.build_expr(ops[1])))

        if fn_name == "_ZN2at18in_parallel_regionEv":
            raise NotImplementedError(f"Unhandled call: _ZN2at18in_parallel_regionEv")

        if fn_name == "_ZN2at20TensorIteratorConfig15add_const_inputERKNS_10TensorBaseE":
            return Concat(self.build_expr(ops[0]), Unit(self.build_expr(ops[1])))

        if fn_name == "_ZN3c10eqEN6caffe28TypeMetaENS_10ScalarTypeE":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZNK2at6Tensor6valuesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6valuesEv")

        if fn_name == "_ZNK3c106detail16integer_iteratorIlLb0ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorIlLb0ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeIlLb0ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIlLb0ELb1EE5beginEv")

        if fn_name == "_ZNK3c1013integer_rangeIlLb0ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIlLb0ELb1EE3endEv")

        if fn_name == "_ZN3c1011SmallVectorIlLj5EEC2INS_8ArrayRefIlEETnNSt9enable_ifIXaasr3stdE16is_convertible_vINSt15iterator_traitsIDTcldtclsr3stdE7declvalIT_EE5beginEEE17iterator_categoryESt18input_iterator_tagEsr3stdE16is_convertible_vINS6_IDTcldtclsr3stdE7declvalIS7_EE3endEEE17iterator_categoryESB_EEiE4typeELi0EEEOS7_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1011SmallVectorIlLj5EEC2INS_8ArrayRefIlEETnNSt9enable_ifIXaasr3stdE16is_convertible_vINSt15iterator_traitsIDTcldtclsr3stdE7declvalIT_EE5beginEEE17iterator_categoryESt18input_iterator_tagEsr3stdE16is_convertible_vINS6_IDTcldtclsr3stdE7declvalIS7_EE3endEEE17iterator_categoryESB_EEiE4typeELi0EEEOS7_")

        if fn_name == "_ZNK3c108ArrayRefIlE5sliceEmm":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE5sliceEmm")

        if fn_name == "_ZN2at27borrow_from_optional_tensorERKSt8optionalINS_6TensorEE":
            raise NotImplementedError(f"Unhandled call: _ZN2at27borrow_from_optional_tensorERKSt8optionalINS_6TensorEE")

        if fn_name == "_ZN3c108ArrayRefIlEC2ISaIlEEERKSt6vectorIlT_E":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK3c108ArrayRefIlE3vecEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at10TensorBase7is_cudaEv":
            return BoolVal(False)

        if fn_name == "_ZN2at20TensorIteratorConfig20check_all_same_dtypeEb":
            # We don't check just return
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at6native22get_nested_tensor_implERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native22get_nested_tensor_implERKNS_6TensorE")

        if fn_name == "_ZNK3c1015SmallVectorImplIlEeqERKS1_":
            return self.build_expr(ops[0]) == self.build_expr(ops[1])

        if fn_name == "_ZNK2at10TensorBase21suggest_memory_formatEb":
            # Suggest 0
            return IntVal(0)

        if fn_name == "_ZN3c1014isIntegralTypeENS_10ScalarTypeEb":
            raise NotImplementedError(f"Unhandled call: _ZN3c1014isIntegralTypeENS_10ScalarTypeEb")

        if fn_name == "_ZNK2at10TensorBase13is_sparse_csrEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase13is_sparse_csrEv")

        if fn_name == "_ZNK3c106detail16integer_iteratorIiLb1ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorIiLb1ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeIiLb1ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIiLb1ELb1EE5beginEv")

        if fn_name == "_ZN3c106irangeIiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_")

        if fn_name == "_ZNK3c1013integer_rangeIiLb1ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIiLb1ELb1EE3endEv")

        if fn_name == "_ZN3c106DeviceC2ENS_10DeviceTypeEa":
            raise NotImplementedError(f"Unhandled call: _ZN3c106DeviceC2ENS_10DeviceTypeEa")

        if fn_name == "_ZN2at20TensorIteratorConfig14resize_outputsEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig14resize_outputsEb")

        if fn_name == "makeConstantArray":
            seq_expr = Empty(bm.IntArray)
            for op in ops:
                seq_expr = Concat(seq_expr, Unit(self.build_expr(op)))
            return seq_expr

        if fn_name == "_ZNK3c108ArrayRefIlE5emptyEv":
            return Length(self.build_expr(ops[0])) == 0

        if fn_name == "_ZN3c1013isComplexTypeENS_10ScalarTypeE":
            stype = self.build_expr(ops[0])
            return And(stype != 8, stype != 9, stype != 10)

        if fn_name == "_ZN3c108ArrayRefIlEC2ERKl":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIlEC2ERKl")

        if fn_name == "_ZN2at10empty_likeERKNS_6TensorEN3c1013TensorOptionsESt8optionalINS3_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10empty_likeERKNS_6TensorEN3c1013TensorOptionsESt8optionalINS3_12MemoryFormatEE")

        if fn_name == "_ZNKSt8optionalIN2at6TensorEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN2at6TensorEE9has_valueEv")

        if fn_name == "_ZNK3c1016OptionalArrayRefIlE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1016OptionalArrayRefIlE9has_valueEv")

        if fn_name == "_ZN3c106irangeIilTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIilTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_")

        if fn_name == "_ZNK3c1013TensorOptions6layoutESt8optionalINS_6LayoutEE":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions6layoutESt8optionalINS_6LayoutEE")

        if fn_name == "_ZNSt17basic_string_viewIcSt11char_traitsIcEEC2EPKc":
            raise NotImplementedError(f"Unhandled call: _ZNSt17basic_string_viewIcSt11char_traitsIcEEC2EPKc")

        if fn_name == "_ZN3c1014isFloatingTypeENS_10ScalarTypeE":
            stype = self.build_expr(ops[0])  # ops[0] is a ScalarType
            return Or(
                stype == 5,  # Half
                stype == 6,  # Float
                stype == 7,  # Double
                stype == 15  # BFloat16
                # add new ones if you care about Float8, etc.
            )

        if fn_name == "_ZNK3c1013TensorOptions13pinned_memoryESt8optionalIbE":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions13pinned_memoryESt8optionalIbE")

        if fn_name == "_ZNK3c1013TensorOptions6deviceESt8optionalINS_6DeviceEE":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions6deviceESt8optionalINS_6DeviceEE")

        if fn_name == "_ZSteqIcSt11char_traitsIcEEbSt17basic_string_viewIT_T0_ENSt15__type_identityIS5_E4typeE":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIcSt11char_traitsIcEEbSt17basic_string_viewIT_T0_ENSt15__type_identityIS5_E4typeE")

        if fn_name == "_ZN3c107canCastENS_10ScalarTypeES0_":
            return BoolVal(True)

        if fn_name == "_ZNK2at18TensorIteratorBase5numelEv":
            raise NotImplementedError("_ZNK2at18TensorIteratorBase5numelEv")

        if fn_name == "_ZNK2at10TensorBase5numelEv":
            tensor_opt = self.build_expr(ops[0])
            sizes = bm.TensorStruct.sizes(tensor_opt)
            numel = IntVal(1)
            for i in range(4):
                idx = IntVal(i)
                # multiply only if i < length(sizes)
                numel = numel * If(idx < Length(sizes), sizes[idx], IntVal(1))
            return numel

        if fn_name == "_ZN3c108ArrayRefIlEC2INS_11SmallVectorIlLj5EEEPlvEERKT_":
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at20TensorIteratorConfig21set_check_mem_overlapEb":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK2at6native16NestedTensorImpl8opt_sizeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6native16NestedTensorImpl8opt_sizeEl")

        if fn_name == "_ZN3c106irangeIllTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIllTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_")

        if fn_name == "_ZN2at10sparse_csr20is_sparse_compressedERKN3c106LayoutE":
            raise NotImplementedError(f"Unhandled call: _ZN2at10sparse_csr20is_sparse_compressedERKN3c106LayoutE")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2IRNS_11SmallVectorIlLj5EEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISA_St10in_place_tEsr3stdE18is_constructible_vINS_8ArrayRefIlEEOS8_Esr3stdE16is_convertible_vISE_SD_Entsr3stdE16is_convertible_vISE_lEEbE4typeELb0EEESE_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2IRNS_11SmallVectorIlLj5EEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISA_St10in_place_tEsr3stdE18is_constructible_vINS_8ArrayRefIlEEOS8_Esr3stdE16is_convertible_vISE_SD_Entsr3stdE16is_convertible_vISE_lEEbE4typeELb0EEESE_")

        if fn_name == "_ZNK3c108ArrayRefIlE6equalsES1_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE6equalsES1_")

        if fn_name == "_ZNK2at6native16NestedTensorImpl16get_nested_sizesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6native16NestedTensorImpl16get_nested_sizesEv")

        if fn_name == "_ZNK2at6native16NestedTensorImpl10get_bufferEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6native16NestedTensorImpl10get_bufferEv")

        if fn_name == "_ZNSaIlEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSaIlEC2Ev")

        if fn_name == "_ZNK3c106detail16integer_iteratorImLb1ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorImLb1ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeImLb1ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeImLb1ELb1EE5beginEv")

        if fn_name == "_ZN3c106irangeImTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeImTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1EEENS_13integer_rangeIS2_Lb1ELb1EEES2_")

        if fn_name == "_ZNK3c1013integer_rangeImLb1ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeImLb1ELb1EE3endEv")

        if fn_name == "_ZNK3c106detail16integer_iteratorIiLb0ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorIiLb0ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeIiLb0ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIiLb0ELb1EE5beginEv")

        if fn_name == "_ZN3c106irangeIiiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIiiTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_")

        if fn_name == "_ZNK3c1013integer_rangeIiLb0ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeIiLb0ELb1EE3endEv")

        if fn_name == "_ZNK3c106detail16integer_iteratorIlLb1ELi0EEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorIlLb1ELi0EEdeEv")

        if fn_name == "_ZNK2at6Tensor9unsqueezeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor9unsqueezeEl")

        if fn_name == "_ZNK2at10TensorBase7is_metaEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase7is_metaEv")

        if fn_name == "_ZNK2at6Tensor7_valuesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7_valuesEv")

        if fn_name == "_ZN3c1014maybe_wrap_dimEllb":
            raise NotImplementedError(f"Unhandled call: _ZN3c1014maybe_wrap_dimEllb")

        if fn_name == "_ZN3c106ScalarC2Ei":
            raise NotImplementedError(f"Unhandled call: _ZN3c106ScalarC2Ei")

        if fn_name == "_ZNR2at6TensoraSERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNR2at6TensoraSERKS0_")

        if fn_name == "_ZNK3c108ArrayRefIN2at6TensorEEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIN2at6TensorEEixEm")

        if fn_name == "_ZN2at14TensorIterator9binary_opERNS_10TensorBaseERKS1_S4_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14TensorIterator9binary_opERNS_10TensorBaseERKS1_S4_")

        if fn_name == "_ZNK2at10TensorBase10ndimensionEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase10ndimensionEv")

        if fn_name == "_ZNK2at18TensorIteratorBase11tensor_baseEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase11tensor_baseEl")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2IRNS_8ArrayRefIlEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISA_St10in_place_tEsr3stdE18is_constructible_vIS4_OS8_Esr3stdE16is_convertible_vISC_S4_Entsr3stdE16is_convertible_vISC_lEEbE4typeELb0EEESC_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2IRNS_8ArrayRefIlEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISA_St10in_place_tEsr3stdE18is_constructible_vIS4_OS8_Esr3stdE16is_convertible_vISC_S4_Entsr3stdE16is_convertible_vISC_lEEbE4typeELb0EEESC_")

        if fn_name == "_ZN3c10eqERKNS_8ArrayRefIlEERKNS_16OptionalArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqERKNS_8ArrayRefIlEERKNS_16OptionalArrayRefIlEE")

        if fn_name == "_ZNR3c1016OptionalArrayRefIlEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNR3c1016OptionalArrayRefIlEdeEv")

        if fn_name == "_ZN2at10sparse_csr12getBlockSizeERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at10sparse_csr12getBlockSizeERKNS_6TensorE")

        if fn_name == "_ZN2at5equalERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at5equalERKNS_6TensorES2_")

        if fn_name == "_ZNK2at6Tensor4viewEN3c108ArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4viewEN3c108ArrayRefIlEE")

        if fn_name == "_ZNSt8optionalIN3c1010ScalarTypeEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c1010ScalarTypeEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_")

        if fn_name == "_ZNSt8optionalIN3c1012MemoryFormatEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c1012MemoryFormatEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_")

        if fn_name == "_ZN2at14TensorIterator8unary_opERNS_10TensorBaseERKS1_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14TensorIterator8unary_opERNS_10TensorBaseERKS1_")

        if fn_name == "_ZNK3c1013TensorOptions6deviceEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions6deviceEv")

        if fn_name == "_ZNK3c106Device6is_cpuEv":
            return BoolVal(True)

        if fn_name == "_ZNSt8optionalIN3c1010ScalarTypeEEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN3c1010ScalarTypeEEC2ESt9nullopt_t")

        if fn_name == "_ZNK2at6Tensor6selectEll":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6selectEll")

        if fn_name == "_ZNK2at10TensorBase12is_quantizedEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase12is_quantizedEv")

        if fn_name == "_ZNK2at6native16NestedTensorImpl18get_nested_stridesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6native16NestedTensorImpl18get_nested_stridesEv")

        if fn_name == "_ZN3c1011SmallVectorIlLj5EEC2EmRKl":
            raise NotImplementedError(f"Unhandled call: _ZN3c1011SmallVectorIlLj5EEC2EmRKl")

        if fn_name == "_ZN9__gnu_cxxneIPN2at6TensorESt6vectorIS2_SaIS2_EEEEbRKNS_17__normal_iteratorIT_T0_EESC_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxneIPN2at6TensorESt6vectorIS2_SaIS2_EEEEbRKNS_17__normal_iteratorIT_T0_EESC_")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EE5beginEv")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EE3endEv")

        if fn_name == "_ZNSaIcEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSaIcEC2Ev")

        if fn_name == "_ZNK2at6Tensor7reshapeEN3c108ArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7reshapeEN3c108ArrayRefIlEE")

        if fn_name == "_ZNKRSt8optionalIN2at6TensorEE5valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN2at6TensorEE5valueEv")

        if fn_name == "_ZNKSt6vectorIlSaIlEE5emptyEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6vectorIlSaIlEE5emptyEv")

        if fn_name == "_ZNK2at10TensorBase11is_alias_ofERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase11is_alias_ofERKS0_")

        if fn_name == "_ZNK3c108ArrayRefIlE4sizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE4sizeEv")

        if fn_name == "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_")

        if fn_name == "_ZN2at6native21wrapped_scalar_tensorERKN3c106ScalarENS1_6DeviceE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native21wrapped_scalar_tensorERKN3c106ScalarENS1_6DeviceE")

        if fn_name == "_ZNK3c106Device4typeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Device4typeEv")

        if fn_name == "llvm.memmove.p0.p0.i64":
            raise NotImplementedError(f"Unhandled call: llvm.memmove.p0.p0.i64")

        if fn_name == "_ZNK2at10TensorBase7stridesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase7stridesEv")

        if fn_name == "_ZNKRSt8optionalIN2at6TensorEE8value_orIS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN2at6TensorEE8value_orIS1_EES1_OT_")

        if fn_name == "_ZN2at6native13is_mixed_typeIJNS_6TensorES2_EEEbRKS2_DpRKT_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native13is_mixed_typeIJNS_6TensorES2_EEEbRKS2_DpRKT_")

        if fn_name == "_ZNSt8optionalIlEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIlEC2ESt9nullopt_t")

        if fn_name == "_ZNSt8optionalIN3c1010ScalarTypeEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c1010ScalarTypeEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_")

        if fn_name == "_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_")

        if fn_name == "_ZSteqIlSaIlEEbRKSt6vectorIT_T0_ES6_":
            raise NotImplementedError(f"Unhandled call: _ZSteqIlSaIlEEbRKSt6vectorIT_T0_ES6_")

        if fn_name == "_ZNK3c1013TensorOptions17has_memory_formatEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions17has_memory_formatEv")

        if fn_name == "_ZN2at6nativeL13is_contiguousERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL13is_contiguousERKNS_6TensorE")

        if fn_name == "_ZSteqIllENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIllENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_")

        if fn_name == "_ZNK3c108ArrayRefIN2at6TensorEE4sizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIN2at6TensorEE4sizeEv")

        if fn_name == "_ZN3c1020guard_size_obliviousEbPKcl":
            raise NotImplementedError(f"Unhandled call: _ZN3c1020guard_size_obliviousEbPKcl")

        if fn_name == "_ZN3c106sym_eqEll":
            raise NotImplementedError(f"Unhandled call: _ZN3c106sym_eqEll")

        if fn_name == "_ZN2at6detail30computeStorageNbytesContiguousEN3c108ArrayRefIlEEmm":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6detail30computeStorageNbytesContiguousEN3c108ArrayRefIlEEmm")

        if fn_name == "_ZNK6caffe28TypeMeta8itemsizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK6caffe28TypeMeta8itemsizeEv")

        if fn_name == "_ZN3c106sym_leEll":
            raise NotImplementedError(f"Unhandled call: _ZN3c106sym_leEll")

        if fn_name == "_ZN2at6native20maybe_convert_symintIlEET_N3c106SymIntE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native20maybe_convert_symintIlEET_N3c106SymIntE")

        if fn_name == "_ZNK3c107Storage10sym_nbytesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c107Storage10sym_nbytesEv")

        if fn_name == "_ZN3c1012WarningUtils14get_warnAlwaysEv":
            raise NotImplementedError(f"Unhandled call: _ZN3c1012WarningUtils14get_warnAlwaysEv")

        if fn_name == "_ZNK2at6Tensor2toEN3c1010ScalarTypeEbbSt8optionalINS1_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6Tensor2toEN3c1010ScalarTypeEbbSt8optionalINS1_12MemoryFormatEE")

        if fn_name == "_ZNK2at10TensorBase6is_xpuEv":
            return BoolVal(False)

        if fn_name == "_ZN3c1013TensorOptionsC2ENS_12MemoryFormatE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1013TensorOptionsC2ENS_12MemoryFormatE")

        if fn_name == "_ZN2at14TensorIterator20borrowing_nullary_opERKNS_10TensorBaseE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at14TensorIterator20borrowing_nullary_opERKNS_10TensorBaseE")

        if fn_name == "_ZNK2at10TensorBase8sym_sizeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase8sym_sizeEl")

        if fn_name == "_ZNSt6vectorIlSaIlEEC2EmRKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEC2EmRKS0_")

        if fn_name == "_ZN9__gnu_cxxneIPSt6vectorIN2at6TensorESaIS3_EES1_IS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESE_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxneIPSt6vectorIN2at6TensorESaIS3_EES1_IS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESE_")

        if fn_name == "_ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EE5beginEv")

        if fn_name == "_ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EEC2Ev")

        if fn_name == "_ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIS_IN2at6TensorESaIS1_EESaIS3_EE3endEv")

        if fn_name == "_ZNK9__gnu_cxx17__normal_iteratorIPSt6vectorIN2at6TensorESaIS3_EES1_IS5_SaIS5_EEEdeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNK9__gnu_cxx17__normal_iteratorIPSt6vectorIN2at6TensorESaIS3_EES1_IS5_SaIS5_EEEdeEv")

        if fn_name == "_ZN2at14TensorIterator19borrowing_binary_opERKNS_10TensorBaseES3_S3_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at14TensorIterator19borrowing_binary_opERKNS_10TensorBaseES3_S3_")

        if fn_name == "_ZNK2at6TensorixEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6TensorixEl")

        if fn_name == "_ZNK2at7Context27checkSparseTensorInvariantsEv":
            return BoolVal(False)

        if fn_name == "_ZNK2at6Tensor12crow_indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12crow_indicesEv")

        if fn_name == "_ZNK2at6Tensor11col_indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor11col_indicesEv")

        if fn_name == "_ZNKSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE9has_valueEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE9has_valueEv")

        if fn_name == "_ZN2at6native19shape_from_dim_maskERKNS_6TensorESt6bitsetILm64EEb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native19shape_from_dim_maskERKNS_6TensorESt6bitsetILm64EEb")

        if fn_name == "_ZN2at11result_typeERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at11result_typeERKNS_6TensorES2_")

        if fn_name == "_ZNK2at10TensorBase9sym_sizesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase9sym_sizesEv")

        if fn_name == "_ZN3c1010isQIntTypeENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1010isQIntTypeENS_10ScalarTypeE")

        if fn_name == "_ZN3c1010isBitsTypeENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1010isBitsTypeENS_10ScalarTypeE")

        if fn_name == "_ZNKSt8optionalIN3c106LayoutEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c106LayoutEE9has_valueEv")

        if fn_name == "_ZN3c106ScalarC2El":
            raise NotImplementedError(f"Unhandled call: _ZN3c106ScalarC2El")

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2ERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2ERKS2_")

        if fn_name == "_ZNSt6vectorIlSaIlEEaSESt16initializer_listIlE":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEaSESt16initializer_listIlE")

        if fn_name == "_ZSt8isfinited":
            raise NotImplementedError(f"Unhandled call: _ZSt8isfinited")

        if fn_name == "_ZNKSt6vectorIN2at6TensorESaIS1_EE4sizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6vectorIN2at6TensorESaIS1_EE4sizeEv")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EEC2EmRKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EEC2EmRKS2_")

        if fn_name == "_ZNSaIN2at6TensorEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSaIN2at6TensorEEC2Ev")

        if fn_name == "_ZNK3c1010TensorImpl5dtypeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1010TensorImpl5dtypeEv")

        if fn_name == "_ZN2at14TensorIteratorC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZN2at14TensorIteratorC2Ev")

        if fn_name == "_ZNK2at18TensorIteratorBase8ntensorsEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase8ntensorsEv")

        if fn_name == "_ZNSt8optionalIN3c106LayoutEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106LayoutEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_")

        if fn_name == "_ZNSt8optionalIN3c106DeviceEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106DeviceEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_")

        if fn_name == "_ZNK3c1013TensorOptions6layoutEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions6layoutEv")

        if fn_name == "_ZNK3c106DeviceneERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106DeviceneERKS0_")

        if fn_name == "_ZNK2at18TensorIteratorBase11input_dtypeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase11input_dtypeEl")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2Ev")

        if fn_name == "_ZN2at10infer_sizeEN3c108ArrayRefIlEES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at10infer_sizeEN3c108ArrayRefIlEES2_")

        if fn_name == "_ZNK2at6Tensor9transposeEll":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor9transposeEll")

        if fn_name == "_ZN2at6native21get_dtype_from_resultERNS_6TensorESt8optionalIN3c1010ScalarTypeEE":
            result_param = self.build_expr(ops[0])
            dtype_optional = self.build_expr(ops[1])
            # if length of dtype_optional is 1, return dtype_optional[0], otherwise, return bm.TensorStruct.dtype(result_param)

            optional_has_value = Length(dtype_optional) == 1

            return If(
                optional_has_value,
                dtype_optional[0],
                bm.TensorStruct.dtype(result_param)
            )

        if fn_name == "_ZN2at6native13is_mixed_typeIJNS_6TensorES2_S2_S2_EEEbRKS2_DpRKT_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native13is_mixed_typeIJNS_6TensorES2_S2_S2_EEEbRKS2_DpRKT_")

        if fn_name == "_ZNK2at6Tensor8coalesceEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8coalesceEv")

        if fn_name == "_ZN2at6native32nested_tensor_impl_is_contiguousEPKNS0_16NestedTensorImplE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native32nested_tensor_impl_is_contiguousEPKNS0_16NestedTensorImplE")

        if fn_name == "_ZNK3c106SymInteqERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106SymInteqERKS0_")

        if fn_name == "_ZN3c108ArrayRefIlEC2EPKlm":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIlEC2EPKlm")

        if fn_name == "_ZNK3c108ArrayRefIlE4dataEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE4dataEv")

        if fn_name == "_ZN3c106ScalarC2Ed":
            raise NotImplementedError(f"Unhandled call: _ZN3c106ScalarC2Ed")

        if fn_name == "_ZN2at20TensorIteratorConfig9add_inputERKNS_10TensorBaseE":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig9add_inputERKNS_10TensorBaseE")

        if fn_name == "_ZNK3c1010MaybeOwnedIN2at6TensorEEptEv":
            return self.build_expr(ops[0])

        if fn_name == "_ZN2at6native7DEFAULTL23reduced_float_type_copyEbRNS_18TensorIteratorBaseE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native7DEFAULTL23reduced_float_type_copyEbRNS_18TensorIteratorBaseE")

        if fn_name == "_ZN2at6native7DEFAULT12_GLOBAL__N_113reduced_inputEN3c1010ScalarTypeES4_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native7DEFAULT12_GLOBAL__N_113reduced_inputEN3c1010ScalarTypeES4_")

        if fn_name == "_ZN2at6native7DEFAULT12_GLOBAL__N_114reduced_outputEN3c1010ScalarTypeES4_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native7DEFAULT12_GLOBAL__N_114reduced_outputEN3c1010ScalarTypeES4_")

        if fn_name == "_ZN2at10TensorBaseC2ERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZN2at10TensorBaseC2ERKS0_")

        if fn_name == "_ZNK2at6Tensor6narrowElll":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6narrowElll")

        if fn_name == "_ZNK2at6Tensor4_nnzEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4_nnzEv")

        if fn_name == "_ZNK2at18TensorIteratorBase9is_scalarEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase9is_scalarEl")

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_")

        if fn_name == "_ZN3c108ArrayRefIN2at6TensorEEC2ERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIN2at6TensorEEC2ERKS2_")

        if fn_name == "_ZN2at6native20review_reduce_resultERKNS_6TensorEiSt6bitsetILm64EEb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native20review_reduce_resultERKNS_6TensorEiSt6bitsetILm64EEb")

        if fn_name == "_ZNK2at6Tensor5cloneESt8optionalIN3c1012MemoryFormatEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor5cloneESt8optionalIN3c1012MemoryFormatEE")

        if fn_name == "_ZNKSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEcvbEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEcvbEv")

        if fn_name == "_ZNKRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEdeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEdeEv")

        if fn_name == "_ZN2at6native41searchsorted_dims_matched_before_last_dimERKNS_6TensorES3_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native41searchsorted_dims_matched_before_last_dimERKNS_6TensorES3_")

        if fn_name == "_ZNKSt8optionalIN3c1010ScalarTypeEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c1010ScalarTypeEE9has_valueEv")

        if fn_name == "_ZNK3c108ArrayRefIN2at6TensorEE5emptyEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIN2at6TensorEE5emptyEv")

        if fn_name == "_ZN2at6native14make_reductionEPKcRNS_6TensorERKS3_N3c1016OptionalArrayRefIlEEbNS7_10ScalarTypeE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native14make_reductionEPKcRNS_6TensorERKS3_N3c1016OptionalArrayRefIlEEbNS7_10ScalarTypeE")

        if fn_name == "_ZNK3c1015SmallVectorBaseIjE4sizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1015SmallVectorBaseIjE4sizeEv")

        if fn_name == "_ZNK3c106detail16integer_iteratorImLb1ELi0EEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorImLb1ELi0EEdeEv")

        if fn_name == "_ZN3c10eqIfEEbRKNS_7complexIT_EES5_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqIfEEbRKNS_7complexIT_EES5_")

        if fn_name == "_ZNK3c106Scalar2toINS_7complexIfEEEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toINS_7complexIfEEEET_v")

        if fn_name == "_ZN3c107complexIfEC2ERKfS3_":
            raise NotImplementedError(f"Unhandled call: _ZN3c107complexIfEC2ERKfS3_")

        if fn_name == "_ZN3c10eqIdEEbRKNS_7complexIT_EES5_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqIdEEbRKNS_7complexIT_EES5_")

        if fn_name == "_ZNK3c106Scalar2toINS_7complexIdEEEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toINS_7complexIdEEEET_v")

        if fn_name == "_ZN3c107complexIdEC2ERKdS3_":
            raise NotImplementedError(f"Unhandled call: _ZN3c107complexIdEC2ERKdS3_")

        if fn_name == "_ZNK3c106Scalar2toIsEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIsEET_v")

        if fn_name == "_ZNK3c106Scalar2toIlEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIlEET_v")

        if fn_name == "_ZNK3c106Scalar2toIiEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIiEET_v")

        if fn_name == "_ZNK3c106Scalar2toIaEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIaEET_v")

        if fn_name == "_ZNK3c106Scalar2toIhEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIhEET_v")

        if fn_name == "_ZNK2at6Tensor5equalERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor5equalERKS0_")

        if fn_name == "_ZNK2at6Tensor2mTEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2mTEv")

        if fn_name == "_ZNK2at6Tensor10as_stridedEN3c108ArrayRefIlEES3_St8optionalIlE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor10as_stridedEN3c108ArrayRefIlEES3_St8optionalIlE")

        if fn_name == "_ZNK3c106detail16integer_iteratorImLb0ELi0EEneERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorImLb0ELi0EEneERKS2_")

        if fn_name == "_ZNK3c1013integer_rangeImLb0ELb1EE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeImLb0ELb1EE5beginEv")

        if fn_name == "_ZN3c106irangeIimTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c106irangeIimTnNSt9enable_ifIXsr3stdE13is_integral_vIT_EEbE4typeELb1ETnNS1_IXsr3stdE13is_integral_vIT0_EEbE4typeELb1EEENS_13integer_rangeIS5_Lb0ELb1EEES2_S5_")

        if fn_name == "_ZNK3c1013integer_rangeImLb0ELb1EE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013integer_rangeImLb0ELb1EE3endEv")

        if fn_name == "_ZSt3getILm0EJN2at6TensorES1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN2at6TensorES1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_")

        if fn_name == "_ZNK2at6Tensor8_indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8_indicesEv")

        if fn_name == "_ZN3c108ArrayRefINS_6SymIntEEC2INS_11SmallVectorIS1_Lj5EEEPS1_vEERKT_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c108ArrayRefINS_6SymIntEEC2INS_11SmallVectorIS1_Lj5EEEPS1_vEERKT_")

        if fn_name == "_ZNK2at6Tensor4itemIbEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4itemIbEET_v")

        if fn_name == "_ZN2at12view_as_realERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at12view_as_realERKNS_6TensorE")

        if fn_name == "_ZNKSt8optionalIN2at6TensorEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN2at6TensorEEcvbEv")

        if fn_name == "_ZN2atL22use_two_pass_reductionERNS_18TensorIteratorBaseE":
            raise NotImplementedError(f"Unhandled call: _ZN2atL22use_two_pass_reductionERNS_18TensorIteratorBaseE")

        if fn_name == "_ZNK2at18TensorIteratorBase6outputEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase6outputEl")

        if fn_name == "_ZN2at14TensorIterator9reduce_opERNS_10TensorBaseERKS1_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14TensorIterator9reduce_opERNS_10TensorBaseERKS1_")

        if fn_name == "_ZNK2at18TensorIteratorBase5inputEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase5inputEl")

        if fn_name == "_ZNSt8optionalIN3c1010ScalarTypeEEC2Ev":
            seq_expr = Empty(SeqSort(IntSort()))
            return seq_expr

        if fn_name == "_ZN3c1020typeMetaToScalarTypeEN6caffe28TypeMetaE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1020typeMetaToScalarTypeEN6caffe28TypeMetaE")

        if fn_name == "_ZNSt6vectorIlSaIlEE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEE5beginEv")

        if fn_name == "_ZNSt6vectorIlSaIlEE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEE3endEv")

        if fn_name == "_ZNK2at10TensorBase14_is_zerotensorEv":
            # We assume we won't feed zerotensor
            return BoolVal(False)

        if fn_name == "_ZNK2at6Tensor3sumEN3c1016OptionalArrayRefIlEEbSt8optionalINS1_10ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6Tensor3sumEN3c1016OptionalArrayRefIlEEbSt8optionalINS1_10ScalarTypeEE")

        if fn_name == "_ZNKRSt8optionalIlE8value_orIiEElOT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIlE8value_orIiEElOT_")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2ERKSt16initializer_listIlE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2ERKSt16initializer_listIlE")

        if fn_name == "_ZNK2at6Tensor6expandEN3c108ArrayRefIlEEb":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6expandEN3c108ArrayRefIlEEb")

        if fn_name == "_ZNK3c1013TensorOptions19merge_memory_formatESt8optionalINS_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK3c1013TensorOptions19merge_memory_formatESt8optionalINS_12MemoryFormatEE")

        if fn_name == "_ZN2at6native27thnn_conv_use_channels_lastERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native27thnn_conv_use_channels_lastERKNS_6TensorES3_")

        if fn_name == "_ZN3c10neIlEEbNS_8ArrayRefIT_EES3_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10neIlEEbNS_8ArrayRefIT_EES3_")

        if fn_name == "_ZNK2at6Tensor9new_emptyEN3c108ArrayRefIlEENS1_13TensorOptionsE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6Tensor9new_emptyEN3c108ArrayRefIlEENS1_13TensorOptionsE")

        if fn_name == "_ZN3c10neIdEEbRKNS_7complexIT_EERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10neIdEEbRKNS_7complexIT_EERKS2_")

        if fn_name == "_ZNK3c106Scalar15toComplexDoubleEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar15toComplexDoubleEv")

        if fn_name == "_ZNK2at7Context36deterministicFillUninitializedMemoryEv":
            return BoolVal(False)

        if fn_name == "_ZNK3c106Scalar10isIntegralEb":
            stype = bm.ScalarStruct.dtype(self.build_expr(ops[0]))
            include_bool = self.build_expr(ops[1])  # second arg is bool

            # Floating types: Half(5), Float(6), Double(7), BFloat16(15)
            is_floating = Or(stype == 5, stype == 6, stype == 7, stype == 15)

            # Bool type = 11
            is_bool = stype == 11

            return If(
                is_floating,
                False,
                If(is_bool, include_bool, True)
            )

        if fn_name == "_ZNKR2at6Tensor17expect_contiguousEN3c1012MemoryFormatE":
            raise NotImplementedError(f"Unhandled call: _ZNKR2at6Tensor17expect_contiguousEN3c1012MemoryFormatE")

        if fn_name == "_ZN2at6native12_GLOBAL__N_124padding_memory_format_3dERKNS_6TensorE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_124padding_memory_format_3dERKNS_6TensorE")

        if fn_name == "_ZNSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEC2ESt9nullopt_t":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEC2ESt9nullopt_t")

        if fn_name == "_ZNKSt6bitsetILm64EE4testEm":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6bitsetILm64EE4testEm")

        if fn_name == "_ZNKRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE5valueEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE5valueEv")

        if fn_name == "_ZSt3getILm1EN2at6TensorES1_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EN2at6TensorES1_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS6_")

        if fn_name == "_ZZN2at6nativeL28sparse_compressed_to_flippedERKNS_6TensorESt8optionalIN3c108ArrayRefIlEEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEENK3$_3clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6nativeL28sparse_compressed_to_flippedERKNS_6TensorESt8optionalIN3c108ArrayRefIlEEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEENK3$_3clEv")

        if fn_name == "_ZNK3c108ArrayRefIlE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE5beginEv")

        if fn_name == "_ZNK3c108IListRefIN2at6TensorEE11materializeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108IListRefIN2at6TensorEE11materializeEv")

        if fn_name == "_ZNKSt17reference_wrapperIKN2at6TensorEEcvRS2_Ev":
            raise NotImplementedError(f"Unhandled call: _ZNKSt17reference_wrapperIKN2at6TensorEEcvRS2_Ev")

        if fn_name == "_ZNSt8optionalIlEC2IlTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES3_IS4_ISt10in_place_tSB_EESt16is_constructibleIlJS7_EESt14is_convertibleIS7_lEEEbE4typeELb1EEEOS7_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIlEC2IlTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES3_IS4_ISt10in_place_tSB_EESt16is_constructibleIlJS7_EESt14is_convertibleIS7_lEEEbE4typeELb1EEEOS7_")

        if fn_name == "_ZNK2at10TensorBase7is_conjEv":
            # Return no conj?
            bm.add_constraint(bm.checked_tensor_conj == True)
            return BoolVal(False)

        if fn_name == "_ZN2at6native17use_mkldnn_matmulERKNS_6TensorES3_S3_":
            return BoolVal(False)

        if fn_name == "_ZNSt6vectorIlSaIlEEC2EmRKlRKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEC2EmRKlRKS0_")

        if fn_name == "_ZNK2at6Tensor7flattenEll":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7flattenEll")

        if fn_name == "_ZN2atneERKNS_6TensorERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZN2atneERKNS_6TensorERKN3c106ScalarE")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2IRSt6vectorIlSaIlEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISB_St10in_place_tEsr3stdE18is_constructible_vINS_8ArrayRefIlEEOS9_Esr3stdE16is_convertible_vISF_SE_Entsr3stdE16is_convertible_vISF_lEEbE4typeELb0EEESF_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2IRSt6vectorIlSaIlEETnNSt9enable_ifIXaaaaaaaantsr3stdE9is_same_vINSt5decayIT_E4typeES1_Entsr3stdE9is_same_vISB_St10in_place_tEsr3stdE18is_constructible_vINS_8ArrayRefIlEEOS9_Esr3stdE16is_convertible_vISF_SE_Entsr3stdE16is_convertible_vISF_lEEbE4typeELb0EEESF_")

        if fn_name == "_ZNK2at6Tensor3sumESt8optionalIN3c1010ScalarTypeEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor3sumESt8optionalIN3c1010ScalarTypeEE")

        if fn_name == "_ZNKSt14_Bit_referencecvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt14_Bit_referencecvbEv")

        if fn_name == "_ZNSt6vectorIbSaIbEEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIbSaIbEEixEm")

        if fn_name == "_ZNK2at18TensorIteratorBase13is_contiguousEv":
            bm.add_constraint(bm.checked_contiguous == True)
            return BoolVal(True)

        if fn_name == "_ZN2at14TensorIteratorC2ERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14TensorIteratorC2ERKS0_")

        if fn_name == "_ZNK3c1013TensorOptions8merge_inES0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions8merge_inES0_")

        if fn_name == "_ZN2at6native23cloneBatchedColumnMajorERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native23cloneBatchedColumnMajorERKNS_6TensorE")

        if fn_name == "_ZN2at5zerosEN3c108ArrayRefIlEENS0_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at5zerosEN3c108ArrayRefIlEENS0_13TensorOptionsE")

        if fn_name == "_ZSt3getILm1EJN2at6TensorES1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EJN2at6TensorES1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_")

        if fn_name == "_ZNK3c1013TensorOptions9has_dtypeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions9has_dtypeEv")

        if fn_name == "_ZN2at11expand_sizeERKNS_6TensorEN3c108ArrayRefIlEEPKc":
            return update_tensor_options_sizes(bm, self.build_expr(ops[0]), self.build_expr(ops[1]))

        if fn_name == "_ZN2at6native19ensure_nonempty_vecESt6vectorIlSaIlEE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native19ensure_nonempty_vecESt6vectorIlSaIlEE")

        if fn_name == "_ZNKSt8optionalIN2at6TensorEEptEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN2at6TensorEEptEv")

        if fn_name == "_ZNKSt8optionalIN3c108ArrayRefIdEEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c108ArrayRefIdEEE9has_valueEv")

        if fn_name == "_ZNSt6vectorIlSaIlEEC2ESt16initializer_listIlERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEC2ESt16initializer_listIlERKS0_")

        if fn_name == "_ZNKRSt8optionalIlE8value_orIlEElOT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIlE8value_orIlEElOT_")

        if fn_name == "_ZNK3c106Device7is_cudaEv":
            return BoolVal(False)

        if fn_name == "_ZN2at6nativeL21options_to_value_typeEN3c1013TensorOptionsE":
            return self.build_expr(ops[0])

        if fn_name == "_ZNK3c106Device6is_xpuEv":
            return BoolVal(False)

        if fn_name == "_ZNSt8optionalIN3c106DeviceEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN3c106DeviceEEC2Ev")

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2IRS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES6_IS7_ISt10in_place_tSE_EESt16is_constructibleIS1_JSA_EESt14is_convertibleISA_S1_EEEbE4typeELb1EEEOSA_")

        if fn_name == "_ZN2at4meanERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS3_10ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4meanERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS3_10ScalarTypeEE")

        if fn_name == "_ZNKSt17reference_wrapperIKN2at6TensorEE3getEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt17reference_wrapperIKN2at6TensorEE3getEv")

        if fn_name == "_ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EEixEm")

        if fn_name == "_ZN2at19legacy_cat_wrap_dimElRKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS4_EE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at19legacy_cat_wrap_dimElRKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS4_EE")

        if fn_name == "_ZN9__gnu_cxxneIPlSt6vectorIlSaIlEEEEbRKNS_17__normal_iteratorIT_T0_EESA_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxneIPlSt6vectorIlSaIlEEEEbRKNS_17__normal_iteratorIT_T0_EESA_")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EEaSEOS3_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EEaSEOS3_")

        if fn_name == "_ZNKSt6vectorISt8optionalIdESaIS1_EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6vectorISt8optionalIdESaIS1_EEixEm")

        if fn_name == "_ZNSt6vectorISt8optionalIdESaIS1_EEC2ESt16initializer_listIS1_ERKS2_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt6vectorISt8optionalIdESaIS1_EEC2ESt16initializer_listIS1_ERKS2_")

        if fn_name == "_ZNSaISt8optionalIdEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSaISt8optionalIdEEC2Ev")

        if fn_name == "_ZN2at3allERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at3allERKNS_6TensorE")

        if fn_name == "_ZN2ateqERKNS_6TensorERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZN2ateqERKNS_6TensorERKN3c106ScalarE")

        if fn_name == "_ZNK3c1013TensorOptions6deviceIJRKNS_10DeviceTypeEEEES0_DpOT_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions6deviceIJRKNS_10DeviceTypeEEEES0_DpOT_")

        if fn_name == "_ZNK2at6Tensor2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6Tensor2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE")

        if fn_name == "_ZN3c1015toRealValueTypeENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1015toRealValueTypeENS_10ScalarTypeE")

        if fn_name == "_ZN2at14TensorIterator14unary_float_opERNS_10TensorBaseERKS1_":
            # Return an array of tensor, the first one is output, the second one is output
            first_tensor = self.build_expr(ops[0])
            second_tensor = self.build_expr(ops[1])
            seq_expr = Empty(SeqSort(bm.TensorStruct))
            # It is incorrect, as the ops[0] size should be different, but on purpose
            seq_expr = Concat(seq_expr, Unit(first_tensor))
            seq_expr = Concat(seq_expr, Unit(second_tensor))
            return seq_expr

        if fn_name == "_ZN2at6native13resize_outputERKNS_6TensorEN3c108ArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native13resize_outputERKNS_6TensorEN3c108ArrayRefIlEE")

        if fn_name == "_ZNK3c108ArrayRefIlE5sliceEm":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE5sliceEm")

        if fn_name == "_ZN2at6native12_GLOBAL__N_114view_weight_2dERKNS_6TensorEN3c1012MemoryFormatE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_114view_weight_2dERKNS_6TensorEN3c1012MemoryFormatE")

        if fn_name == "_ZNKSt8optionalIlEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIlEcvbEv")

        if fn_name == "_ZNSt8optionalIlEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIlEC2Ev")

        if fn_name == "_ZNK3c106Scalar9isComplexEv":
            stype = bm.TensorStruct.dtype(self.build_expr(ops[0]))
            return And(stype != 8, stype != 9, stype != 10)

        if fn_name == "_ZNK3c106Scalar9isBooleanEv":
            stype = bm.ScalarStruct.dtype(self.build_expr(ops[0]))
            # Bool dtype is 11
            return stype == 11

        if fn_name == "_ZNKSt8optionalIN3c106ScalarEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c106ScalarEEcvbEv")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EEixEm")

        if fn_name == "_ZN3c108GradMode10is_enabledEv":
            raise NotImplementedError(f"Unhandled call: _ZN3c108GradMode10is_enabledEv")

        if fn_name == "_ZNK2at10TensorBase13requires_gradEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase13requires_gradEv")

        if fn_name == "_ZNK3c108ArrayRefINS_6SymIntEE6equalsES2_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefINS_6SymIntEE6equalsES2_")

        if fn_name == "_ZN2at23infer_size_symdimvectorEN3c108ArrayRefINS0_6SymIntEEES3_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at23infer_size_symdimvectorEN3c108ArrayRefINS0_6SymIntEEES3_")

        if fn_name == "_ZN12_GLOBAL__N_119is_supported_deviceEN3c106DeviceE":
            raise NotImplementedError(f"Unhandled call: _ZN12_GLOBAL__N_119is_supported_deviceEN3c106DeviceE")

        if fn_name == "_ZN3c10eqENS_10ScalarTypeEN6caffe28TypeMetaE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqENS_10ScalarTypeEN6caffe28TypeMetaE")

        if fn_name == "_ZN6caffe28TypeMeta4MakeIN3c108quint2x4EEES0_v":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe28TypeMeta4MakeIN3c108quint2x4EEES0_v")

        if fn_name == "_ZNK2at6Tensor25q_per_channel_zero_pointsEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor25q_per_channel_zero_pointsEv")

        if fn_name == "_ZN6caffe28TypeMeta4MakeIN3c108quint4x2EEES0_v":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe28TypeMeta4MakeIN3c108quint4x2EEES0_v")

        if fn_name == "_ZN6caffe28TypeMeta4MakeIN3c106qint32EEES0_v":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe28TypeMeta4MakeIN3c106qint32EEES0_v")

        if fn_name == "_ZN6caffe28TypeMeta4MakeIN3c106quint8EEES0_v":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe28TypeMeta4MakeIN3c106quint8EEES0_v")

        if fn_name == "_ZN6caffe28TypeMeta4MakeIN3c105qint8EEES0_v":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe28TypeMeta4MakeIN3c105qint8EEES0_v")

        if fn_name == "_ZN12_GLOBAL__N_120copy_transpose_validERKN2at6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN12_GLOBAL__N_120copy_transpose_validERKN2at6TensorES3_")

        if fn_name == "_ZN2at20TensorIteratorConfig21check_all_same_deviceEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig21check_all_same_deviceEb")

        if fn_name == "_ZNK2at6Tensor2geERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2geERKN3c106ScalarE")

        if fn_name == "_ZNK2at6Tensor3minEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor3minEv")

        if fn_name == "_ZNK2at6native16NestedTensorImpl28get_unsafe_storage_as_tensorEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6native16NestedTensorImpl28get_unsafe_storage_as_tensorEv")

        if fn_name == "_ZNSt8optionalIbEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIbEC2ESt9nullopt_t")

        if fn_name == "_ZNK2at10TensorBase14storage_offsetEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase14storage_offsetEv")

        if fn_name == "_ZN2at6nativeeqElNS0_16EmbeddingBagModeE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeeqElNS0_16EmbeddingBagModeE")

        if fn_name == "_ZN2at6hasMKLEv":
            return BoolVal(False)

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2ESt9nullopt_t")

        if fn_name == "_ZN2at18dim_list_to_bitsetEN3c1016OptionalArrayRefIlEEm":
            raise NotImplementedError(f"Unhandled call: _ZN2at18dim_list_to_bitsetEN3c1016OptionalArrayRefIlEEm")

        if fn_name == "_ZN2at20TensorIteratorConfig30promote_inputs_to_common_dtypeEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig30promote_inputs_to_common_dtypeEb")

        if fn_name == "_ZN2at32_convert_indices_from_csr_to_cooERKNS_6TensorES2_bb":
            raise NotImplementedError(f"Unhandled call: _ZN2at32_convert_indices_from_csr_to_cooERKNS_6TensorES2_bb")

        if fn_name == "_ZSt3getILm0EN2at6TensorES1_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EN2at6TensorES1_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS6_")

        if fn_name == "_ZN2at10sparse_csr25getCompressedPlainIndicesERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at10sparse_csr25getCompressedPlainIndicesERKNS_6TensorE")

        if fn_name == "_ZNKRSt8optionalIbE8value_orIbEEbOT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIbE8value_orIbEEbOT_")

        if fn_name == "_ZNK2at6Tensor3cpuEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor3cpuEv")

        if fn_name == "_ZN3c10neIlEEbNS_8ArrayRefIT_EERKSt6vectorIS2_SaIS2_EE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10neIlEEbNS_8ArrayRefIT_EERKSt6vectorIS2_SaIS2_EE")

        if fn_name == "_ZNK2at6Tensor13to_sparse_csrESt8optionalIlE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor13to_sparse_csrESt8optionalIlE")

        if fn_name == "_ZN3c1011SmallVectorIlLj5EEC2IPKlvEET_S5_":
            raise NotImplementedError(f"Unhandled call: _ZN3c1011SmallVectorIlLj5EEC2IPKlvEET_S5_")

        if fn_name == "_ZNK3c108ArrayRefIlE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE3endEv")

        if fn_name == "_ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE4sizeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE4sizeEv")

        if fn_name == "_ZNK3c1013TensorOptions10device_optEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions10device_optEv")

        if fn_name == "_ZNK2at6Tensor6unbindEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6unbindEl")

        if fn_name == "_ZNSt6vectorIlSaIlEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEC2Ev")

        if fn_name == "_ZN3c1013toComplexTypeENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1013toComplexTypeENS_10ScalarTypeE")

        if fn_name == "_ZNK3c106Scalar6toBoolEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar6toBoolEv")

        if fn_name == "_ZNK2at6Tensor4itemEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4itemEv")

        if fn_name == "_ZNK2at6Tensor5sliceElSt8optionalIlES2_l":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor5sliceElSt8optionalIlES2_l")

        if fn_name == "_ZN3c1013TensorOptionsC2IJRKNS_10DeviceTypeEEvEEDpOT_.specialized.4":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1013TensorOptionsC2IJRKNS_10DeviceTypeEEvEEDpOT_.specialized.4")

        if fn_name == "_ZNSaIbEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSaIbEC2Ev")

        if fn_name == "_ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c105qint8ESt5arrayINS1_10VectorizedIfEELm4EES5_INS6_INS3_6qint32EEELm4EELi32EE14float_num_vecsEv":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c105qint8ESt5arrayINS1_10VectorizedIfEELm4EES5_INS6_INS3_6qint32EEELm4EELi32EE14float_num_vecsEv")

        if fn_name == "_ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c106quint8ESt5arrayINS1_10VectorizedIfEELm4EES5_INS6_INS3_6qint32EEELm4EELi32EE14float_num_vecsEv":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c106quint8ESt5arrayINS1_10VectorizedIfEELm4EES5_INS6_INS3_6qint32EEELm4EELi32EE14float_num_vecsEv")

        if fn_name == "_ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c106qint32ESt5arrayINS1_10VectorizedIfEELm1EES5_INS6_IS4_EELm1EELi8EE14float_num_vecsEv":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at3vec7DEFAULT28VectorizedQuantizedConverterIN3c106qint32ESt5arrayINS1_10VectorizedIfEELm1EES5_INS6_IS4_EELm1EELi8EE14float_num_vecsEv")

        if fn_name == "_ZNK3c1013TensorOptions13memory_formatESt8optionalINS_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK3c1013TensorOptions13memory_formatESt8optionalINS_12MemoryFormatEE")

        if fn_name == "_ZNK2at10TensorBase7storageEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase7storageEv")

        if fn_name == "_ZN2at6native10is_nonzeroERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native10is_nonzeroERKNS_6TensorE")

        if fn_name == "_ZSt3getILm0EJN2at6TensorES1_EERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN2at6TensorES1_EERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS6_")

        if fn_name == "_ZNSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEC2IRKS3_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS4_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES9_ISA_ISt10in_place_tSH_EESt16is_constructibleIS3_JSD_EESt14is_convertibleISD_S3_EEEbE4typeELb1EEEOSD_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEC2IRKS3_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS4_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES9_ISA_ISt10in_place_tSH_EESt16is_constructibleIS3_JSD_EESt14is_convertibleISD_S3_EEEbE4typeELb1EEEOSD_")

        if fn_name == "_ZNSt8optionalIN3c106ScalarEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106ScalarEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_")

        if fn_name == "_ZNK2at6Tensor10sparse_dimEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor10sparse_dimEv")

        if fn_name == "_ZNK2at18TensorIteratorBase6tensorEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at18TensorIteratorBase6tensorEl")

        if fn_name == "_ZN2at6native12_GLOBAL__N_138make_value_selection_intersection_iterERKNS_6TensorES4_S4_S4_S4_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_138make_value_selection_intersection_iterERKNS_6TensorES4_S4_S4_S4_")

        if fn_name == "_ZN2at6arangeERKN3c106ScalarENS0_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6arangeERKN3c106ScalarENS0_13TensorOptionsE")

        if fn_name == "_ZNSt8optionalIN3c106DeviceEEC2IRKNS0_10DeviceTypeETnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES8_IS9_ISt10in_place_tSG_EESt16is_constructibleIS1_JSC_EESt14is_convertibleISC_S1_EEEbE4typeELb1EEEOSC_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106DeviceEEC2IRKNS0_10DeviceTypeETnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES8_IS9_ISt10in_place_tSG_EESt16is_constructibleIS1_JSC_EESt14is_convertibleISC_S1_EEEbE4typeELb1EEEOSC_")

        if fn_name == "_ZN2at6native26searchsorted_scalar_tensorERKN3c106ScalarERKNS1_6DeviceE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native26searchsorted_scalar_tensorERKN3c106ScalarERKNS1_6DeviceE")

        if fn_name == "_ZN3c1012promoteTypesENS_10ScalarTypeES0_":
            raise NotImplementedError(f"Unhandled call: _ZN3c1012promoteTypesENS_10ScalarTypeES0_")

        if fn_name == "_ZNKSt8optionalIN3c106LayoutEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c106LayoutEEcvbEv")

        if fn_name == "_ZNSt8optionalIN3c106LayoutEEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN3c106LayoutEEC2ESt9nullopt_t")

        if fn_name == "_ZNSt8optionalIN3c106DeviceEEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN3c106DeviceEEC2ESt9nullopt_t")

        if fn_name == "_ZNKRSt8optionalIN2at6TensorEEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN2at6TensorEEdeEv")

        if fn_name == "_ZN3c10eqINS_6SymIntEEEbNS_8ArrayRefIT_EES4_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqINS_6SymIntEEEbNS_8ArrayRefIT_EES4_")

        if fn_name == "_ZNSt6bitsetILm64EEC2Ey":
            raise NotImplementedError(f"Unhandled call: _ZNSt6bitsetILm64EEC2Ey")

        if fn_name == "_ZN3c1017asIntArrayRefSlowENS_8ArrayRefINS_6SymIntEEEPKcl":
            raise NotImplementedError(f"Unhandled call: _ZN3c1017asIntArrayRefSlowENS_8ArrayRefINS_6SymIntEEEPKcl")

        if fn_name == "_ZSt3getILm0EJN3c1011SmallVectorINS0_6SymIntELj5EEES3_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN3c1011SmallVectorINS0_6SymIntELj5EEES3_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_")

        if fn_name == "_ZN2at6nativeL20inferSqueezeGeometryERKNS_6TensorESt6bitsetILm64EE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL20inferSqueezeGeometryERKNS_6TensorESt6bitsetILm64EE")

        if fn_name == "_ZSt3getILm0EN3c1010MaybeOwnedIN2at6TensorEEES4_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EN3c1010MaybeOwnedIN2at6TensorEEES4_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS9_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_124promoteIndicesAndOffsetsERKNS_6TensorES4_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_124promoteIndicesAndOffsetsERKNS_6TensorES4_")

        if fn_name == "_ZN2at20TensorIteratorConfig28cast_common_dtype_to_outputsEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig28cast_common_dtype_to_outputsEb")

        if fn_name == "_ZNKSt8optionalIdEcvbEv":
            return Length(self.build_expr(ops[0])) > 0

        if fn_name == "_ZN2at4meta26make_reduction_from_out_tyERKNS_6TensorES3_N3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4meta26make_reduction_from_out_tyERKNS_6TensorES3_N3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE")

        if fn_name == "_ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EEixEm")

        if fn_name == "_ZN9__gnu_cxxneIPKSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESG_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxneIPKSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESG_")

        if fn_name == "_ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5beginEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5beginEv")

        if fn_name == "_ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE3endEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE3endEv")

        if fn_name == "_ZN2at6nativeL18_batch_tile_tensorERKNS_6TensorEN3c108ArrayRefIlEEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL18_batch_tile_tensorERKNS_6TensorEN3c108ArrayRefIlEEl")

        if fn_name == "_ZNK2at6Tensor3mulERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor3mulERKN3c106ScalarE")

        if fn_name == "_ZN2at6native26is_std_inner_dim_fast_pathERKNS_6TensorEN3c1016OptionalArrayRefIlEERKSt8optionalINS4_6ScalarEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native26is_std_inner_dim_fast_pathERKNS_6TensorEN3c1016OptionalArrayRefIlEERKSt8optionalINS4_6ScalarEE")

        if fn_name == "_ZN2at6native10out_deviceIJKNS_6TensorES2_S2_EEEN3c106DeviceEDpRT_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native10out_deviceIJKNS_6TensorES2_S2_EEEN3c106DeviceEDpRT_")

        if fn_name == "_ZNK3c1010TensorImpl3dimEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1010TensorImpl3dimEv")

        if fn_name == "_ZNKSt6bitsetILm64EE3anyEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6bitsetILm64EE3anyEv")

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_")

        if fn_name == "_ZN2at3catERKN3c108IListRefINS_6TensorEEEl":
            raise NotImplementedError(f"Unhandled call: _ZN2at3catERKN3c108IListRefINS_6TensorEEEl")

        if fn_name == "_ZN3c10eqERKNS_6SymIntEi":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqERKNS_6SymIntEi")

        if fn_name == "_ZN2at6native12_GLOBAL__N_118HelperInterpLinear28compute_index_ranges_weightsEN3c1010ScalarTypeElllllbRKSt8optionalIdEb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_118HelperInterpLinear28compute_index_ranges_weightsEN3c1010ScalarTypeElllllbRKSt8optionalIdEb")

        if fn_name == "_ZNK2at6Tensor4conjEv":
            bm.add_constraint(bm.checked_tensor_conj == True)
            return self.build_expr(ops[0])

        if fn_name == "_ZSt3getILm1EJN3c1010MaybeOwnedIN2at6TensorEEES4_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EJN3c1010MaybeOwnedIN2at6TensorEEES4_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_")

        if fn_name == "_ZN2at15expand_outplaceERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at15expand_outplaceERKNS_6TensorES2_")

        if fn_name == "_ZSt3getILm0EJN3c1010MaybeOwnedIN2at6TensorEEES4_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN3c1010MaybeOwnedIN2at6TensorEEES4_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_")

        if fn_name == "_ZN2at6native8internal12_GLOBAL__N_112all_positiveERN3c108ArrayRefIlEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native8internal12_GLOBAL__N_112all_positiveERN3c108ArrayRefIlEE")

        if fn_name == "_ZN2at6native8internal12_GLOBAL__N_115all_nonnegativeERSt6vectorIlSaIlEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native8internal12_GLOBAL__N_115all_nonnegativeERSt6vectorIlSaIlEE")

        if fn_name == "_ZN3c10eqIlEEbNS_8ArrayRefIT_EERKSt6vectorIS2_SaIS2_EE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqIlEEbNS_8ArrayRefIT_EERKSt6vectorIS2_SaIS2_EE")

        if fn_name == "_ZN2at6native14make_reductionEPKcRNS_6TensorES4_RKS3_N3c1016OptionalArrayRefIlEEbNS7_10ScalarTypeE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native14make_reductionEPKcRNS_6TensorES4_RKS3_N3c1016OptionalArrayRefIlEEbNS7_10ScalarTypeE")

        if fn_name == "_ZStneIcSt11char_traitsIcEEbSt17basic_string_viewIT_T0_ENSt15__type_identityIS5_E4typeE":
            raise NotImplementedError(
                f"Unhandled call: _ZStneIcSt11char_traitsIcEEbSt17basic_string_viewIT_T0_ENSt15__type_identityIS5_E4typeE")

        if fn_name == "_ZNKRSt8optionalIN3c1010ScalarTypeEE8value_orIS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN3c1010ScalarTypeEE8value_orIS1_EES1_OT_")

        if fn_name == "_ZN2at7_uniqueERKNS_6TensorEbb":
            raise NotImplementedError(f"Unhandled call: _ZN2at7_uniqueERKNS_6TensorEbb")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIlhEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIlhEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at6tensorEN3c108ArrayRefIlEERKNS0_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6tensorEN3c108ArrayRefIlEERKNS0_13TensorOptionsE")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIlaEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIlaEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIliEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIliEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIllEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIllEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIlsEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIlsEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIddEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIddEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIffEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIffEESt5tupleIJNS_6TensorES3_EEN3c1013TensorOptionsENS5_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIN3c107complexIdEES4_EESt5tupleIJNS_6TensorES6_EENS2_13TensorOptionsENS2_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIN3c107complexIdEES4_EESt5tupleIJNS_6TensorES6_EENS2_13TensorOptionsENS2_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIN3c107complexIfEES4_EESt5tupleIJNS_6TensorES6_EENS2_13TensorOptionsENS2_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIN3c107complexIfEES4_EESt5tupleIJNS_6TensorES6_EENS2_13TensorOptionsENS2_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIfN3c104HalfEEESt5tupleIJNS_6TensorES5_EENS2_13TensorOptionsENS2_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIfN3c104HalfEEESt5tupleIJNS_6TensorES5_EENS2_13TensorOptionsENS2_10ScalarTypeEl")

        if fn_name == "_ZN2at10sparse_csr17create_acc_bufferIfN3c108BFloat16EEESt5tupleIJNS_6TensorES5_EENS2_13TensorOptionsENS2_10ScalarTypeEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr17create_acc_bufferIfN3c108BFloat16EEESt5tupleIJNS_6TensorES5_EENS2_13TensorOptionsENS2_10ScalarTypeEl")

        if fn_name == "_ZSt3getILm0EJN2at6TensorES1_St8optionalIS1_EEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN2at6TensorES1_St8optionalIS1_EEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_")

        if fn_name == "_ZN2at6nativeL38sparse_mask_like_prepare_sparse_inputsERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_6TensorESB_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL38sparse_mask_like_prepare_sparse_inputsERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_6TensorESB_")

        if fn_name == "_ZSt3getILm1EJN2at6TensorES1_St8optionalIS1_EEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EJN2at6TensorES1_St8optionalIS1_EEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS8_")

        if fn_name == "_ZNK2at6Tensor1tEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor1tEv")

        if fn_name == "_ZN2at6native30get_nested_tensor_impl_or_nullERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native30get_nested_tensor_impl_or_nullERKNS_6TensorE")

        if fn_name == "_ZN2at6native18get_reduction_enumERKSt17basic_string_viewIcSt11char_traitsIcEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native18get_reduction_enumERKSt17basic_string_viewIcSt11char_traitsIcEE")

        if fn_name == "_ZN2at11expand_sizeERKNS_6TensorEN3c108ArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZN2at11expand_sizeERKNS_6TensorEN3c108ArrayRefIlEE")

        if fn_name == "_ZN3c10eqIdEEbRKNS_7complexIT_EERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZN3c10eqIdEEbRKNS_7complexIT_EERKS2_")

        if fn_name == "_ZSteqIN3c106LayoutES1_ENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS3_ES8_":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIN3c106LayoutES1_ENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS3_ES8_")

        if fn_name == "_ZN2at6TensorC2EONS_10TensorBaseE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6TensorC2EONS_10TensorBaseE")

        if fn_name == "_ZN2at6detail9empty_cpuEN3c108ArrayRefIlEESt8optionalINS1_10ScalarTypeEES4_INS1_6LayoutEES4_INS1_6DeviceEES4_IbES4_INS1_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6detail9empty_cpuEN3c108ArrayRefIlEESt8optionalINS1_10ScalarTypeEES4_INS1_6LayoutEES4_INS1_6DeviceEES4_IbES4_INS1_12MemoryFormatEE")

        if fn_name == "_ZN2at6native9empty_cpuEN3c108ArrayRefIlEESt8optionalINS1_10ScalarTypeEES4_INS1_6LayoutEES4_INS1_6DeviceEES4_IbES4_INS1_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native9empty_cpuEN3c108ArrayRefIlEESt8optionalINS1_10ScalarTypeEES4_INS1_6LayoutEES4_INS1_6DeviceEES4_IbES4_INS1_12MemoryFormatEE")

        if fn_name == "_ZN3c1023optTypeMetaToScalarTypeESt8optionalIN6caffe28TypeMetaEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1023optTypeMetaToScalarTypeESt8optionalIN6caffe28TypeMetaEE")

        if fn_name == "_ZNK3c1013TensorOptions9dtype_optEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions9dtype_optEv")

        if fn_name == "_ZNK3c1013TensorOptions10layout_optEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions10layout_optEv")

        if fn_name == "_ZNK3c1013TensorOptions17pinned_memory_optEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions17pinned_memory_optEv")

        if fn_name == "_ZNK3c106Scalar2toIdEET_v":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar2toIdEET_v")

        if fn_name == "_ZNSt8optionalIlEC2IRlTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES4_IS5_ISt10in_place_tSC_EESt16is_constructibleIlJS8_EESt14is_convertibleIS8_lEEEbE4typeELb1EEEOS8_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIlEC2IRlTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES4_IS5_ISt10in_place_tSC_EESt16is_constructibleIlJS8_EESt14is_convertibleIS8_lEEEbE4typeELb1EEEOS8_")

        if fn_name == "_ZN2at6native17borrow_else_cloneEbRKNS_6TensorES3_b":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native17borrow_else_cloneEbRKNS_6TensorES3_b")

        if fn_name == "_ZNSt8optionalIN2at6TensorEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN2at6TensorEEC2Ev")

        if fn_name == "_ZSt5isinfd":
            raise NotImplementedError(f"Unhandled call: _ZSt5isinfd")

        if fn_name == "_ZSt5isnand":
            raise NotImplementedError(f"Unhandled call: _ZSt5isnand")

        if fn_name == "_ZNK3c1010MaybeOwnedIN2at10TensorBaseEEptEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1010MaybeOwnedIN2at10TensorBaseEEptEv")

        if fn_name == "_ZN2at14expand_inplaceERKNS_10TensorBaseES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14expand_inplaceERKNS_10TensorBaseES2_")

        if fn_name == "_ZNK2at10TensorBase2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at10TensorBase2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE")

        if fn_name == "_ZN3c1013TensorOptionsC2IJRKNS_10DeviceTypeEEvEEDpOT_.specialized.4.27098":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1013TensorOptionsC2IJRKNS_10DeviceTypeEEvEEDpOT_.specialized.4.27098")

        if fn_name == "_ZN2at6native12_GLOBAL__N_111get_offsetsERKNS_6TensorERKN3c108ArrayRefIlEEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_111get_offsetsERKNS_6TensorERKN3c108ArrayRefIlEEl")

        if fn_name == "_ZN9__gnu_cxxmiIPlSt6vectorIlSaIlEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxmiIPlSt6vectorIlSaIlEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_")

        if fn_name == "_ZSt11lower_boundIN9__gnu_cxx17__normal_iteratorIPlSt6vectorIlSaIlEEEElET_S7_S7_RKT0_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt11lower_boundIN9__gnu_cxx17__normal_iteratorIPlSt6vectorIlSaIlEEEElET_S7_S7_RKT0_")

        if fn_name == "_ZNSt6vectorIlSaIlEEixEm":
            arr, idx = self.build_expr(ops[0]), self.build_expr(ops[1])
            return arr[idx]

        if fn_name == "_ZNKSt8optionalIN3c1011SmallVectorIlLj5EEEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c1011SmallVectorIlLj5EEEE9has_valueEv")

        if fn_name == "_ZN2at6detail13computeStrideEN3c108ArrayRefIlEES3_RKNS1_11SmallVectorIlLj5EEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6detail13computeStrideEN3c108ArrayRefIlEES3_RKNS1_11SmallVectorIlLj5EEE")

        if fn_name == "_ZN2at13infer_size_dvEN3c108ArrayRefIlEEl":
            raise NotImplementedError(f"Unhandled call: _ZN2at13infer_size_dvEN3c108ArrayRefIlEEl")

        if fn_name == "_ZN2at6native28_linalg_broadcast_batch_dimsERKNS_6TensorES3_PKc":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native28_linalg_broadcast_batch_dimsERKNS_6TensorES3_PKc")

        if fn_name == "_ZN2at6native19resize_output_checkERKNS_6TensorEN3c108ArrayRefIlEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native19resize_output_checkERKNS_6TensorEN3c108ArrayRefIlEE")

        if fn_name == "_ZN2at6native27is_row_or_column_contiguousERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native27is_row_or_column_contiguousERKNS_6TensorE")

        if fn_name == "_ZN2at6TensorC2EOS0_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6TensorC2EOS0_")

        if fn_name == "_ZNK2at10TensorBase6is_negEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase6is_negEv")

        if fn_name == "_ZNK2at6Tensor11resolve_negEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor11resolve_negEv")

        if fn_name == "_ZN2at6native27is_mean_inner_dim_fast_pathERKNS_6TensorEN3c1016OptionalArrayRefIlEESt8optionalINS4_10ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native27is_mean_inner_dim_fast_pathERKNS_6TensorEN3c1016OptionalArrayRefIlEESt8optionalINS4_10ScalarTypeEE")

        if fn_name == "_ZNK2at6Tensor4add_ERKS0_RKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4add_ERKS0_RKN3c106ScalarE")

        if fn_name == "_ZNK2at6Tensor8squeeze_El":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8squeeze_El")

        if fn_name == "_ZNK2at6Tensor10transpose_Ell":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor10transpose_Ell")

        if fn_name == "_ZN3c108ArrayRefIN2at6TensorEEC2ISaIS2_EEERKSt6vectorIS2_T_E":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIN2at6TensorEEC2ISaIS2_EEERKSt6vectorIS2_T_E")

        if fn_name == "_ZN2at6nativeL26allocate_bin_edges_tensorsERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL26allocate_bin_edges_tensorsERKNS_6TensorE")

        if fn_name == "_ZSt10accumulateIPKllSt10multipliesIlEET0_T_S5_S4_T1_":
            raise NotImplementedError(f"Unhandled call: _ZSt10accumulateIPKllSt10multipliesIlEET0_T_S5_S4_T1_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_112solve_arangeERKNS_6TensorERlS5_S5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_112solve_arangeERKNS_6TensorERlS5_S5_")

        if fn_name == "_ZN2at5whereERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at5whereERKNS_6TensorE")

        if fn_name == "_ZNK2at6Tensor2eqERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2eqERKN3c106ScalarE")

        if fn_name == "_ZNK3c106Scalar5equalIdTnNSt9enable_ifIXntsr3c1010is_complexIT_EE5valueEiE4typeELi0EEEbS3_":
            raise NotImplementedError(
                f"Unhandled call: _ZNK3c106Scalar5equalIdTnNSt9enable_ifIXntsr3c1010is_complexIT_EE5valueEiE4typeELi0EEEbS3_")

        if fn_name == "_ZN2at6native18maybe_native_stackERNS_6TensorEN3c108ArrayRefIS1_EEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native18maybe_native_stackERNS_6TensorEN3c108ArrayRefIS1_EEl")

        if fn_name == "_ZNK3c106detail16integer_iteratorImLb0ELi0EEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106detail16integer_iteratorImLb0ELi0EEdeEv")

        if fn_name == "_ZN6caffe2neERKNS_8TypeMetaES2_":
            raise NotImplementedError(f"Unhandled call: _ZN6caffe2neERKNS_8TypeMetaES2_")

        if fn_name == "_ZN2at6native6detail23CanUseNativeSerialStackIN3c108ArrayRefINS_6TensorEEELb0EE4callERS5_S6_l":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native6detail23CanUseNativeSerialStackIN3c108ArrayRefINS_6TensorEEELb0EE4callERS5_S6_l")

        if fn_name == "_ZN2at6native10empty_likeERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native10empty_likeERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE")

        if fn_name == "_ZN2at6native10zeros_likeERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native10zeros_likeERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE")

        if fn_name == "_ZSteqIllENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT0_EEclsr3stdE7declvalIRKT_EEEbEE5valueEbE4typeES3_RKSt8optionalIS4_E":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIllENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT0_EEclsr3stdE7declvalIRKT_EEEbEE5valueEbE4typeES3_RKSt8optionalIS4_E")

        if fn_name == "_ZNK3c108ArrayRefIlEixEm":
            arr, idx = self.build_expr(ops[0]), self.build_expr(ops[1])
            return arr[idx]

        if fn_name == "_ZN2at6sparse14is_same_tensorERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6sparse14is_same_tensorERKNS_6TensorES3_")

        if fn_name == "_ZN2at4meta14make_reductionERKNS_6TensorES3_N3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4meta14make_reductionERKNS_6TensorES3_N3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE")

        if fn_name == "_ZNKRSt8optionalIN3c106LayoutEE8value_orIRKS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN3c106LayoutEE8value_orIRKS1_EES1_OT_")

        if fn_name == "_ZN2at6nativeL15build_addr_iterERNS_6TensorERKS1_S4_S4_":
            # Special implementation
            seq_expr = Empty(SeqSort(bm.TensorStruct))
            # It is incorrect, as the ops[0] size should be different, but on purpose
            seq_expr = Concat(seq_expr, Unit(self.build_expr(ops[1])))
            seq_expr = Concat(seq_expr, Unit(self.build_expr(ops[1])))
            seq_expr = Concat(seq_expr, Unit(self.build_expr(ops[2])))
            seq_expr = Concat(seq_expr, Unit(self.build_expr(ops[3])))
            return seq_expr

        if fn_name == "_ZNK2at6Tensor12index_selectElRKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12index_selectElRKS0_")

        if fn_name == "_ZNSt6vectorIbSaIbEEC2EmRKbRKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIbSaIbEEC2EmRKbRKS0_")

        if fn_name == "_ZNK2at6Tensor3allEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor3allEv")

        if fn_name == "_ZSt3getILm1EN3c1010MaybeOwnedIN2at6TensorEEES4_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EN3c1010MaybeOwnedIN2at6TensorEEES4_EONSt13tuple_elementIXT_ESt4pairIT0_T1_EE4typeEOS9_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_131is_fast_path_index_select_scaleIlEEbRKNS_6TensorES5_RS3_T_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_131is_fast_path_index_select_scaleIlEEbRKNS_6TensorES5_RS3_T_")

        if fn_name == "_ZN2at6nativeL13make_bag_sizeERKNS_6TensorES3_lbb":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL13make_bag_sizeERKNS_6TensorES3_lbb")

        if fn_name == "_ZN2at6nativeL15make_offset2bagERNS_6TensorERKS1_S4_S4_lRKSt8optionalIS1_El":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL15make_offset2bagERNS_6TensorERKS1_S4_S4_lRKSt8optionalIS1_El")

        if fn_name == "_ZN2at6native12_GLOBAL__N_125is_fast_path_index_selectIlEEbRKNS_6TensorERS3_T_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_125is_fast_path_index_selectIlEEbRKNS_6TensorERS3_T_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_131is_fast_path_index_select_scaleIiEEbRKNS_6TensorES5_RS3_T_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_131is_fast_path_index_select_scaleIiEEbRKNS_6TensorES5_RS3_T_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_125is_fast_path_index_selectIiEEbRKNS_6TensorERS3_T_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_125is_fast_path_index_selectIiEEbRKNS_6TensorERS3_T_")

        if fn_name == "_ZNK3c106Scalar8toDoubleEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar8toDoubleEv")

        if fn_name == "_ZNK3c108ArrayRefIlE8allMatchERKSt8functionIFbRKlEE":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIlE8allMatchERKSt8functionIFbRKlEE")

        if fn_name == "_ZN3c104implneERKNS0_12ListIteratorISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEESG_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c104implneERKNS0_12ListIteratorISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEESG_")

        if fn_name == "_ZNK3c104ListISt8optionalIN2at6TensorEEE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c104ListISt8optionalIN2at6TensorEEE5beginEv")

        if fn_name == "_ZNK3c104ListISt8optionalIN2at6TensorEEE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c104ListISt8optionalIN2at6TensorEEE3endEv")

        if fn_name == "_ZNK3c104impl20ListElementReferenceISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEcvS5_Ev":
            raise NotImplementedError(
                f"Unhandled call: _ZNK3c104impl20ListElementReferenceISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEcvS5_Ev")

        if fn_name == "_ZNK3c104impl12ListIteratorISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEdeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNK3c104impl12ListIteratorISt8optionalIN2at6TensorEEN9__gnu_cxx17__normal_iteratorIPNS_6IValueESt6vectorIS8_SaIS8_EEEEEdeEv")

        if fn_name == "_ZN2at16is_expandable_toEN3c108ArrayRefIlEES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at16is_expandable_toEN3c108ArrayRefIlEES2_")

        if fn_name == "_ZN2at6native9make_infoENS_6TensorEN3c108IListRefINS_17OptionalTensorRefEEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native9make_infoENS_6TensorEN3c108IListRefINS_17OptionalTensorRefEEE")

        if fn_name == "_ZN3c108IListRefIN2at17OptionalTensorRefEEC2ERKNS_4ListISt8optionalINS1_6TensorEEEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c108IListRefIN2at17OptionalTensorRefEEC2ERKNS_4ListISt8optionalINS1_6TensorEEEE")

        if fn_name == "_ZNK3c1016IListRefIteratorIN2at17OptionalTensorRefEEneERKS3_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1016IListRefIteratorIN2at17OptionalTensorRefEEneERKS3_")

        if fn_name == "_ZNK3c108IListRefIN2at17OptionalTensorRefEE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108IListRefIN2at17OptionalTensorRefEE5beginEv")

        if fn_name == "_ZNK3c108IListRefIN2at17OptionalTensorRefEE3endEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108IListRefIN2at17OptionalTensorRefEE3endEv")

        if fn_name == "_ZNK2at17OptionalTensorRef9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at17OptionalTensorRef9has_valueEv")

        if fn_name == "_ZNK3c1016IListRefIteratorIN2at17OptionalTensorRefEEdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1016IListRefIteratorIN2at17OptionalTensorRefEEdeEv")

        if fn_name == "_ZNKR2at17OptionalTensorRefptEv":
            raise NotImplementedError(f"Unhandled call: _ZNKR2at17OptionalTensorRefptEv")

        if fn_name == "_ZNKR2at17OptionalTensorRefdeEv":
            raise NotImplementedError(f"Unhandled call: _ZNKR2at17OptionalTensorRefdeEv")

        if fn_name == "_ZSt8isfinitef":
            raise NotImplementedError(f"Unhandled call: _ZSt8isfinitef")

        if fn_name == "_ZNK3c108BFloat16cvfEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108BFloat16cvfEv")

        if fn_name == "_ZNK3c104HalfcvfEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c104HalfcvfEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_117HelperInterpCubic28compute_index_ranges_weightsEN3c1010ScalarTypeElllllbRKSt8optionalIdEb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_117HelperInterpCubic28compute_index_ranges_weightsEN3c1010ScalarTypeElllllbRKSt8optionalIdEb")

        if fn_name == "_ZN2at6native13is_mixed_typeIJNS_6TensorES2_S2_S2_S2_EEEbRKS2_DpRKT_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native13is_mixed_typeIJNS_6TensorES2_S2_S2_S2_EEEbRKS2_DpRKT_")

        if fn_name == "_ZN2at6nativeL28suggest_memory_format_contigERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL28suggest_memory_format_contigERKNS_6TensorE")

        if fn_name == "_ZN2at2mmERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at2mmERKNS_6TensorES2_")

        if fn_name == "_ZN2at6nativeL24apply_mkldnn_matmul_heurElll":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL24apply_mkldnn_matmul_heurElll")

        if fn_name == "_ZNK2at6Tensor12resolve_conjEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12resolve_conjEv")

        if fn_name == "_ZN2at6native17is_floating_pointERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native17is_floating_pointERKNS_6TensorE")

        if fn_name == "_ZN2at6native10is_complexERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native10is_complexERKNS_6TensorE")

        if fn_name == "_ZZN2at6nativeL19bmm_out_or_baddbmm_ERKNS_6TensorES3_S3_RKN3c106ScalarES7_bENKUlS3_E_clES3_":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6nativeL19bmm_out_or_baddbmm_ERKNS_6TensorES3_S3_RKN3c106ScalarES7_bENKUlS3_E_clES3_")

        if fn_name == "_ZNRSt8optionalIN2at6TensorEE5valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNRSt8optionalIN2at6TensorEE5valueEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_128is_valid_quantization_schemeERKNS_6TensorE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_128is_valid_quantization_schemeERKNS_6TensorE")

        if fn_name == "_ZNK2at6Tensor7q_scaleEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7q_scaleEv")

        if fn_name == "_ZNK2at6Tensor12q_zero_pointEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12q_zero_pointEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_121is_cat_nhwc_fast_pathERKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS6_EEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_121is_cat_nhwc_fast_pathERKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS6_EEl")

        if fn_name == "_ZN2at23_empty_affine_quantizedEN3c108ArrayRefIlEENS0_13TensorOptionsEdlSt8optionalINS0_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at23_empty_affine_quantizedEN3c108ArrayRefIlEENS0_13TensorOptionsEdlSt8optionalINS0_12MemoryFormatEE")

        if fn_name == "_ZNKSt6vectorIlSaIlEE4sizeEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6vectorIlSaIlEE4sizeEv")

        if fn_name == "_ZNSt6vectorIbSaIbEEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIbSaIbEEC2Ev")

        if fn_name == "_ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5emptyEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNKSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5emptyEv")

        if fn_name == "_ZN2at10sparse_csr20is_sparse_compressedERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at10sparse_csr20is_sparse_compressedERKNS_6TensorE")

        if fn_name == "_ZNK3c1013TensorOptions10type_equalERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions10type_equalERKS0_")

        if fn_name == "_ZN2at4fullEN3c108ArrayRefIlEERKNS0_6ScalarENS0_13TensorOptionsE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4fullEN3c108ArrayRefIlEERKNS0_6ScalarENS0_13TensorOptionsE")

        if fn_name == "_ZNSt6vectorIlSaIlEEC2ERKS1_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIlSaIlEEC2ERKS1_")

        if fn_name == "_ZN2at20TensorIteratorConfig30enforce_safe_casting_to_outputEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig30enforce_safe_casting_to_outputEb")

        if fn_name == "_ZNK2at10TensorBase6is_mpsEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase6is_mpsEv")

        if fn_name == "_ZN2at14namedinference15are_names_equalEPN3c1010TensorImplES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at14namedinference15are_names_equalEPN3c1010TensorImplES3_")

        if fn_name == "_ZNK2at6Tensor12is_same_sizeERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12is_same_sizeERKS0_")

        if fn_name == "_ZN2at20TensorIteratorConfig17allow_cpu_scalarsEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig17allow_cpu_scalarsEb")

        if fn_name == "_ZNKR3c1016OptionalArrayRefIlE8value_orINS_11SmallVectorIlLj5EEEEENSt9enable_ifIXsr3stdE16is_convertible_vIOT_NS_8ArrayRefIlEEEES9_E4typeES7_":
            raise NotImplementedError(
                f"Unhandled call: _ZNKR3c1016OptionalArrayRefIlE8value_orINS_11SmallVectorIlLj5EEEEENSt9enable_ifIXsr3stdE16is_convertible_vIOT_NS_8ArrayRefIlEEEES9_E4typeES7_")

        if fn_name == "_ZN3c1011SmallVectorIlLj5EEC2ESt16initializer_listIlE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1011SmallVectorIlLj5EEC2ESt16initializer_listIlE")

        if fn_name == "_ZN2at8can_castEN3c1010ScalarTypeES1_":
            raise NotImplementedError(f"Unhandled call: _ZN2at8can_castEN3c1010ScalarTypeES1_")

        if fn_name == "_ZNK2at6Tensor8mvlgammaEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8mvlgammaEl")

        if fn_name == "_ZN2at6native12_fft_r2c_mklERKNS_6TensorEN3c108ArrayRefIlEElb":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native12_fft_r2c_mklERKNS_6TensorEN3c108ArrayRefIlEElb")

        if fn_name == "_ZNK2at17OptionalTensorRefcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at17OptionalTensorRefcvbEv")

        if fn_name == "_ZNK3c1013TensorOptions10has_layoutEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions10has_layoutEv")

        if fn_name == "_ZN3c108IListRefIN2at6TensorEEC2IJRSt6vectorIS2_SaIS2_EEEvEEDpOT_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c108IListRefIN2at6TensorEEC2IJRSt6vectorIS2_SaIS2_EEEvEEDpOT_")

        if fn_name == "_ZNSt6vectorIN2at6TensorESaIS1_EEC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN2at6TensorESaIS1_EEC2Ev")

        if fn_name == "_ZN2at6nativeL18sizes_match_exceptEN3c108ArrayRefIlEES3_l":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL18sizes_match_exceptEN3c108ArrayRefIlEES3_l")

        if fn_name == "_ZN2at6native17max_quantized_cpuERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native17max_quantized_cpuERKNS_6TensorE")

        if fn_name == "_ZN2at4_ops3sum4callERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4_ops3sum4callERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE")

        if fn_name == "_ZNK3c108ArrayRefINS_6SymIntEE5emptyEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefINS_6SymIntEE5emptyEv")

        if fn_name == "_ZNK3c108ArrayRefINS_6SymIntEEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefINS_6SymIntEEixEm")

        if fn_name == "_ZN3c10rmERKNS_6SymIntEi":
            raise NotImplementedError(f"Unhandled call: _ZN3c10rmERKNS_6SymIntEi")

        if fn_name == "_ZNK2at10TensorBase18sym_storage_offsetEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase18sym_storage_offsetEv")

        if fn_name == "_ZNK2at10TensorBase11sym_stridesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase11sym_stridesEv")

        if fn_name == "_ZN3c1011SmallVectorINS_6SymIntELj5EEC2EmRKS1_":
            raise NotImplementedError(f"Unhandled call: _ZN3c1011SmallVectorINS_6SymIntELj5EEC2EmRKS1_")

        if fn_name == "_ZN3c106SymIntC2Ev":
            raise NotImplementedError(f"Unhandled call: _ZN3c106SymIntC2Ev")

        if fn_name == "_ZNK3c1011OptionalRefINS_6ScalarEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1011OptionalRefINS_6ScalarEEcvbEv")

        if fn_name == "_ZZN2at6native29threshold_backward_sparse_outERKNS_6TensorES3_RKN3c106ScalarERS1_ENK3$_0clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native29threshold_backward_sparse_outERKNS_6TensorES3_RKN3c106ScalarERS1_ENK3$_0clEv")

        if fn_name == "_ZNSt6vectorIbSaIbEEC2EmRKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIbSaIbEEC2EmRKS0_")

        if fn_name == "_ZN2at32_convert_indices_from_coo_to_csrERKNS_6TensorElb":
            raise NotImplementedError(f"Unhandled call: _ZN2at32_convert_indices_from_coo_to_csrERKNS_6TensorElb")

        if fn_name == "_ZNK2at6Tensor7indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7indicesEv")

        if fn_name == "_ZNK2at6Tensor2toEN3c106DeviceEN6caffe28TypeMetaEbb":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2toEN3c106DeviceEN6caffe28TypeMetaEbb")

        if fn_name == "_ZNSt8optionalIN3c106LayoutEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106LayoutEEC2IRKS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES7_IS8_ISt10in_place_tSF_EESt16is_constructibleIS1_JSB_EESt14is_convertibleISB_S1_EEEbE4typeELb1EEEOSB_")

        if fn_name == "_ZNK2at6Tensor12ccol_indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor12ccol_indicesEv")

        if fn_name == "_ZNK2at6Tensor11row_indicesEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor11row_indicesEv")

        if fn_name == "_ZN2at6native8internal15get_output_sizeILl3EEESt6vectorIlSaIlEERKNS_6TensorEN3c108ArrayRefIlEESB_SB_SB_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native8internal15get_output_sizeILl3EEESt6vectorIlSaIlEERKNS_6TensorEN3c108ArrayRefIlEESB_SB_SB_")

        if fn_name == "_ZNK2at6Tensor6cumsumElSt8optionalIN3c1010ScalarTypeEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6cumsumElSt8optionalIN3c1010ScalarTypeEE")

        if fn_name == "_ZNK2at6Tensor6argminESt8optionalIlEb":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor6argminESt8optionalIlEb")

        if fn_name == "_ZN3c108IListRefIN2at6TensorEEC2ERKSt16initializer_listIS2_E":
            raise NotImplementedError(f"Unhandled call: _ZN3c108IListRefIN2at6TensorEEC2ERKSt16initializer_listIS2_E")

        if fn_name == "_ZNSt8optionalIlEC2IiTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES3_IS4_ISt10in_place_tSB_EESt16is_constructibleIlJS7_EESt14is_convertibleIS7_lEEEbE4typeELb1EEEOS7_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIlEC2IiTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES3_IS4_ISt10in_place_tSB_EESt16is_constructibleIlJS7_EESt14is_convertibleIS7_lEEEbE4typeELb1EEEOS7_")

        if fn_name == "_ZNK3c1013TensorOptions10has_deviceEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions10has_deviceEv")

        if fn_name == "_ZN3c107StorageC2EOS0_":
            raise NotImplementedError(f"Unhandled call: _ZN3c107StorageC2EOS0_")

        if fn_name == "_ZNK3c107Storage11is_alias_ofERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c107Storage11is_alias_ofERKS0_")

        if fn_name == "_ZNK3c107Storage6deviceEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c107Storage6deviceEv")

        if fn_name == "_ZN2at10sparse_csr7to_typeERKNS_6TensorEN3c1010ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN2at10sparse_csr7to_typeERKNS_6TensorEN3c1010ScalarTypeE")

        if fn_name == "_ZN2at6native12_fft_c2c_mklERKNS_6TensorEN3c108ArrayRefIlEElb":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native12_fft_c2c_mklERKNS_6TensorEN3c108ArrayRefIlEElb")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9LhsProjOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9LhsProjOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9LhsProjOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9LhsProjOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_119_csr_matmult_maxnnzINS0_21StridedRandomAccessorIllNS0_16DefaultPtrTraitsEEEEElllT_S6_S6_S6_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_119_csr_matmult_maxnnzINS0_21StridedRandomAccessorIllNS0_16DefaultPtrTraitsEEEEElllT_S6_S6_S6_")

        if fn_name == "_ZN2at6native21StridedRandomAccessorIllNS0_16DefaultPtrTraitsEEC2EPll":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native21StridedRandomAccessorIllNS0_16DefaultPtrTraitsEEC2EPll")

        if fn_name == "_ZNK2at10TensorBase8data_ptrIlEEPT_v":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase8data_ptrIlEEPT_v")

        if fn_name == "_ZNK2at10TensorBase6strideEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase6strideEl")

        if fn_name == "_ZN2at6native12_GLOBAL__N_114linear_for_ffnERKNS_6TensorES4_S4_St8optionalIbE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_114linear_for_ffnERKNS_6TensorES4_S4_St8optionalIbE")

        if fn_name == "_ZNSt8optionalIbEC2IRbTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES4_IS5_ISt10in_place_tSC_EESt16is_constructibleIbJS8_EESt14is_convertibleIS8_bEEEbE4typeELb1EEEOS8_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIbEC2IRbTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS0_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES4_IS5_ISt10in_place_tSC_EESt16is_constructibleIbJS8_EESt14is_convertibleIS8_bEEEbE4typeELb1EEEOS8_")

        if fn_name == "_ZN3c108ArrayRefIlEC2EPKlS3_":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIlEC2EPKlS3_")

        if fn_name == "_ZNK2at10TensorBase14const_data_ptrIlTnNSt9enable_ifIXntsr3stdE10is_const_vIT_EEiE4typeELi0EEEPKS3_v":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at10TensorBase14const_data_ptrIlTnNSt9enable_ifIXntsr3stdE10is_const_vIT_EEiE4typeELi0EEEPKS3_v")

        if fn_name == "_ZN2at12_GLOBAL__N_122structured_mul_out_outC2ERNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at12_GLOBAL__N_122structured_mul_out_outC2ERNS_6TensorE")

        if fn_name == "_ZNK3c107SymBool11expect_trueEPKcl":
            raise NotImplementedError(f"Unhandled call: _ZNK3c107SymBool11expect_trueEPKcl")

        if fn_name == "_ZNK3c106SymInt6sym_geERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106SymInt6sym_geERKS0_")

        if fn_name == "_ZN3c106SymIntC2El":
            raise NotImplementedError(f"Unhandled call: _ZN3c106SymIntC2El")

        if fn_name == "_ZNK3c107SymBool7sym_andERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c107SymBool7sym_andERKS0_")

        if fn_name == "_ZNK3c106SymInt6sym_leERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106SymInt6sym_leERKS0_")

        if fn_name == "_ZN3c10ngERKNS_6SymIntE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10ngERKNS_6SymIntE")

        if fn_name == "_ZN3c10ltERKNS_6SymIntEi":
            raise NotImplementedError(f"Unhandled call: _ZN3c10ltERKNS_6SymIntEi")

        if fn_name == "_ZNK3c106SymIntmiERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106SymIntmiERKS0_")

        if fn_name == "_ZN2at6nativeL20can_cat_nested_sizesERKNS_6TensorES3_l":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL20can_cat_nested_sizesERKNS_6TensorES3_l")

        if fn_name == "_ZN2at14maybe_wrap_dimElN3c108ArrayRefINS_6TensorEEE":
            raise NotImplementedError(f"Unhandled call: _ZN2at14maybe_wrap_dimElN3c108ArrayRefINS_6TensorEEE")

        if fn_name == "_ZNK3c1013TensorOptions5dtypeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions5dtypeEv")

        if fn_name == "_ZNK2at6Tensor7qschemeEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7qschemeEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_5MulOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_5MulOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_5MulOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_5MulOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9RhsProjOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9RhsProjOpEEElLl8EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9RhsProjOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_142_sparse_binary_op_intersection_kernel_implINS1_17CPUKernelLauncherENS1_35CPUValueSelectionIntersectionKernelINS1_9RhsProjOpEEElLl0EEEvRNS_6TensorERKS7_SA_RKSt6vectorIlSaIlEERKSt8optionalIS7_ESJ_bbENKUlvE3_clEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_116_allocate_bufferERKNS_6TensorEib":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native12_GLOBAL__N_116_allocate_bufferERKNS_6TensorEib")

        if fn_name == "_ZN2atmlERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2atmlERKNS_6TensorES2_")

        if fn_name == "_ZN2at12index_selectERKNS_6TensorElS2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at12index_selectERKNS_6TensorElS2_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_126_move_memory_if_cuda_inputERKNS_6TensorES4_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_126_move_memory_if_cuda_inputERKNS_6TensorES4_")

        if fn_name == "_ZNK2at6Tensor7squeezeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7squeezeEl")

        if fn_name == "_ZNK2at6Tensor7nonzeroEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7nonzeroEv")

        if fn_name == "_ZN2atgeERKNS_6TensorERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZN2atgeERKNS_6TensorERKN3c106ScalarE")

        if fn_name == "_ZN2at6native12_GLOBAL__N_115operator_1_normERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native12_GLOBAL__N_115operator_1_normERKNS_6TensorE")

        if fn_name == "_ZN3c106ScalarC2Ef":
            raise NotImplementedError(f"Unhandled call: _ZN3c106ScalarC2Ef")

        if fn_name == "_ZN2at3powERKN3c106ScalarERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at3powERKN3c106ScalarERKNS_6TensorE")

        if fn_name == "_ZNK2at6TensorngEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6TensorngEv")

        if fn_name == "_ZNK2at6Tensor5clampERKSt8optionalIN3c106ScalarEES6_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor5clampERKSt8optionalIN3c106ScalarEES6_")

        if fn_name == "_ZN2at4ceilERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at4ceilERKNS_6TensorE")

        if fn_name == "_ZN2at4log2ERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at4log2ERKNS_6TensorE")

        if fn_name == "_ZN2atdvERKNS_6TensorERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZN2atdvERKNS_6TensorERKN3c106ScalarE")

        if fn_name == "_ZNSt8optionalIN3c106ScalarEEC2IiTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106ScalarEEC2IiTnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_")

        if fn_name == "_ZNSt8optionalIN3c106ScalarEEC2ESt9nullopt_t":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN3c106ScalarEEC2ESt9nullopt_t")

        if fn_name == "_ZN2at6native27_compute_linear_combinationERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native27_compute_linear_combinationERKNS_6TensorES3_")

        if fn_name == "_ZN2at9from_blobEPvN3c108ArrayRefIlEES3_RKNS1_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at9from_blobEPvN3c108ArrayRefIlEES3_RKNS1_13TensorOptionsE")

        if fn_name == "_ZN3c1013TensorOptionsC2ENS_10ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN3c1013TensorOptionsC2ENS_10ScalarTypeE")

        if fn_name == "_ZN2at6native21histogramdd_bin_edgesERKNS_6TensorEN3c108ArrayRefIlEESt8optionalINS5_IdEEERKS7_IS1_Eb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native21histogramdd_bin_edgesERKNS_6TensorEN3c108ArrayRefIlEESt8optionalINS5_IdEEERKS7_IS1_Eb")

        if fn_name == "_ZZN2at6native24select_sparse_csr_workerILb0ELb1EEENS_6TensorERKS2_llENKUlvE0_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native24select_sparse_csr_workerILb0ELb1EEENS_6TensorERKS2_llENKUlvE0_clEv")

        if fn_name == "_ZNKRSt8optionalIN3c1012MemoryFormatEE8value_orIS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN3c1012MemoryFormatEE8value_orIS1_EES1_OT_")

        if fn_name == "_ZZN2at6native28structured_index_add_cpu_out4implERKNS_6TensorElS4_S4_RKN3c106ScalarES4_ENK3$_0clENS5_8ArrayRefIlEESB_l":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native28structured_index_add_cpu_out4implERKNS_6TensorElS4_S4_RKN3c106ScalarES4_ENK3$_0clENS5_8ArrayRefIlEESB_l")

        if fn_name == "_ZN3c1010MaybeOwnedIN2at6TensorEE5ownedEOS2_":
            raise NotImplementedError(f"Unhandled call: _ZN3c1010MaybeOwnedIN2at6TensorEE5ownedEOS2_")

        if fn_name == "_ZNK2at6Tensor16to_padded_tensorEdN3c1016OptionalArrayRefIlEE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor16to_padded_tensorEdN3c1016OptionalArrayRefIlEE")

        if fn_name == "_ZN2at6matmulERKNS_6TensorES2_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6matmulERKNS_6TensorES2_")

        if fn_name == "_ZN2at6sparse15is_same_densityERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6sparse15is_same_densityERKNS_6TensorES3_")

        if fn_name == "_ZN2at6native8internal15get_output_sizeILl2EEESt6vectorIlSaIlEERKNS_6TensorEN3c108ArrayRefIlEESB_SB_SB_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native8internal15get_output_sizeILl2EEESt6vectorIlSaIlEERKNS_6TensorEN3c108ArrayRefIlEESB_SB_SB_")

        if fn_name == "_ZSteqIliENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSteqIliENSt9enable_ifIXsr14is_convertibleIDTeqclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_")

        if fn_name == "_ZStneIllENSt9enable_ifIXsr14is_convertibleIDTneclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_":
            raise NotImplementedError(
                f"Unhandled call: _ZStneIllENSt9enable_ifIXsr14is_convertibleIDTneclsr3stdE7declvalIRKT_EEclsr3stdE7declvalIRKT0_EEEbEE5valueEbE4typeERKSt8optionalIS1_ES6_")

        if fn_name == "_ZN2at6native26linalg_solve_is_vector_rhsERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native26linalg_solve_is_vector_rhsERKNS_6TensorES3_")

        if fn_name == "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_":
            raise NotImplementedError(f"Unhandled call: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_")

        if fn_name == "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ISt17basic_string_viewIcS2_EvEERKT_RKS3_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ISt17basic_string_viewIcS2_EvEERKT_RKS3_")

        if fn_name == "_ZNRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE5valueEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE5valueEv")

        if fn_name == "_ZNKSt8optionalIdE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIdE9has_valueEv")

        if fn_name == "_ZStneIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_":
            raise NotImplementedError(
                f"Unhandled call: _ZStneIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_")

        if fn_name == "_ZN2at6nativeL24get_default_lstsq_driverB5cxx11ESt8optionalISt17basic_string_viewIcSt11char_traitsIcEEERKNS_6TensorE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL24get_default_lstsq_driverB5cxx11ESt8optionalISt17basic_string_viewIcSt11char_traitsIcEEERKNS_6TensorE")

        if fn_name == "_ZNK3c1013TensorOptions13pinned_memoryEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions13pinned_memoryEv")

        if fn_name == "_ZZN2at6native24select_sparse_csr_workerILb1ELb0EEENS_6TensorERKS2_llENKUlvE0_clEv":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native24select_sparse_csr_workerILb1ELb0EEENS_6TensorERKS2_llENKUlvE0_clEv")

        if fn_name == "_ZNKSt8optionalIN3c108ArrayRefIdEEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c108ArrayRefIdEEEcvbEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_130enable_qnnpack_for_ada_avgpoolERKNS_6TensorEN3c108ArrayRefIlEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_130enable_qnnpack_for_ada_avgpoolERKNS_6TensorEN3c108ArrayRefIlEE")

        if fn_name == "_ZN3c108ArrayRefIlEC2ILm2EEERKSt5arrayIlXT_EE":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefIlEC2ILm2EEERKSt5arrayIlXT_EE")

        if fn_name == "_ZNK2at6Tensor2gtERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2gtERKN3c106ScalarE")

        if fn_name == "_ZNK2at6Tensor2leERKN3c106ScalarE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor2leERKN3c106ScalarE")

        if fn_name == "_ZNK3c106ScalarngEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106ScalarngEv")

        if fn_name == "_ZN2at6native13is_mixed_typeIJNS_6TensorEEEEbRKS2_DpRKT_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native13is_mixed_typeIJNS_6TensorEEEEbRKS2_DpRKT_")

        if fn_name == "_ZNK2at6Tensor7argsortElb":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor7argsortElb")

        if fn_name == "_ZN3c1010MaybeOwnedIN2at6TensorEEaSEOS3_":
            raise NotImplementedError(f"Unhandled call: _ZN3c1010MaybeOwnedIN2at6TensorEEaSEOS3_")

        if fn_name == "_ZN2at6native6sparse4impl19_is_sparse_and_zeroERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native6sparse4impl19_is_sparse_and_zeroERKNS_6TensorE")

        if fn_name == "_ZN2at18linalg_vector_normERKNS_6TensorERKN3c106ScalarENS3_16OptionalArrayRefIlEEbSt8optionalINS3_10ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at18linalg_vector_normERKNS_6TensorERKN3c106ScalarENS3_16OptionalArrayRefIlEEbSt8optionalINS3_10ScalarTypeEE")

        if fn_name == "_ZN3c1016OptionalArrayRefIlEC2ERKl":
            raise NotImplementedError(f"Unhandled call: _ZN3c1016OptionalArrayRefIlEC2ERKl")

        if fn_name == "_ZNK3c106Scalar5equalEb":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar5equalEb")

        if fn_name == "_ZNSt8functionIFbRKlEEC2IZN2at6native12_GLOBAL__N_122check_maxpool3d_paramsEN3c108ArrayRefIlEESA_SA_SA_E3$_0vEEOT_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8functionIFbRKlEEC2IZN2at6native12_GLOBAL__N_122check_maxpool3d_paramsEN3c108ArrayRefIlEESA_SA_SA_E3$_0vEEOT_")

        if fn_name == "_ZN2at6native17min_quantized_cpuERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native17min_quantized_cpuERKNS_6TensorE")

        if fn_name == "_ZN2at4_ops4prod4callERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4_ops4prod4callERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE")

        if fn_name == "_ZN2at6native12_GLOBAL__N_123make_index_put_iteratorERKNS0_13AdvancedIndexERKNS_6TensorE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_123make_index_put_iteratorERKNS0_13AdvancedIndexERKNS_6TensorE")

        if fn_name == "_ZN3c10gtERKNS_6SymIntEi":
            raise NotImplementedError(f"Unhandled call: _ZN3c10gtERKNS_6SymIntEi")

        if fn_name == "_ZN3c1017multiply_integersISt6vectorINS_6SymIntESaIS2_EETnNSt9enable_ifIXsr3stdE9is_same_vINT_10value_typeES2_EEiE4typeELi0EEES2_RKS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1017multiply_integersISt6vectorINS_6SymIntESaIS2_EETnNSt9enable_ifIXsr3stdE9is_same_vINT_10value_typeES2_EEiE4typeELi0EEES2_RKS6_")

        if fn_name == "_ZN2at17infer_size_symintEN3c108ArrayRefINS0_6SymIntEEES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at17infer_size_symintEN3c108ArrayRefINS0_6SymIntEEES3_")

        if fn_name == "_ZN3c108ArrayRefINS_6SymIntEEC2EPKS1_m":
            raise NotImplementedError(f"Unhandled call: _ZN3c108ArrayRefINS_6SymIntEEC2EPKS1_m")

        if fn_name == "_ZNK3c108ArrayRefINS_6SymIntEE4dataEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefINS_6SymIntEE4dataEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIdLN3c1010ScalarTypeE4EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIdLN3c1010ScalarTypeE4EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_")

        if fn_name == "_ZN3c1028get_contiguous_memory_formatEv":
            raise NotImplementedError(f"Unhandled call: _ZN3c1028get_contiguous_memory_formatEv")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIdLN3c1010ScalarTypeE3EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIdLN3c1010ScalarTypeE3EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIfLN3c1010ScalarTypeE4EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIfLN3c1010ScalarTypeE4EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_")

        if fn_name == "_ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIfLN3c1010ScalarTypeE3EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_":
            raise NotImplementedError(
                f"Unhandled call: _ZZN2at6native12_GLOBAL__N_130ctc_loss_backward_cpu_templateIfLN3c1010ScalarTypeE3EEENS_6TensorERKS5_S7_S7_NS3_8ArrayRefIlEES9_S7_S7_lbENKUlS7_S9_E_clES7_S9_")

        if fn_name == "_ZN2at6native26get_zero_numel_tensor_sizeERKNS_6TensorElbPKc":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native26get_zero_numel_tensor_sizeERKNS_6TensorElbPKc")

        if fn_name == "_ZN2at6native34_dimreduce_return_trivial_no_identERNS_6TensorERKS1_lbPKc":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native34_dimreduce_return_trivial_no_identERNS_6TensorERKS1_lbPKc")

        if fn_name == "_ZNRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEdeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNRSt8optionalISt17basic_string_viewIcSt11char_traitsIcEEEdeEv")

        if fn_name == "_ZN2at6native12_GLOBAL__N_122inferUnsqueezeGeometryERKNS_6TensorEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_122inferUnsqueezeGeometryERKNS_6TensorEl")

        if fn_name == "_ZN2at6linearERKNS_6TensorES2_RKSt8optionalIS0_E":
            raise NotImplementedError(f"Unhandled call: _ZN2at6linearERKNS_6TensorES2_RKSt8optionalIS0_E")

        if fn_name == "_ZN2at10sparse_csr40only_sparse_compressed_add_trivial_casesERKNS_6TensorES3_RKN3c106ScalarERS1_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at10sparse_csr40only_sparse_compressed_add_trivial_casesERKNS_6TensorES3_RKN3c106ScalarERS1_")

        if fn_name == "_ZN2at6native14qkv_projectionERKNS_6TensorES3_S3_lS3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native14qkv_projectionERKNS_6TensorES3_S3_lS3_")

        if fn_name == "_ZN2at6native16split_with_sizesERKNS_6TensorEN3c108ArrayRefIlEEl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native16split_with_sizesERKNS_6TensorEN3c108ArrayRefIlEEl")

        if fn_name == "_ZN2at6native5chunkERKNS_6TensorEll":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native5chunkERKNS_6TensorEll")

        if fn_name == "_ZN2at6detail11make_tensorINS_11QTensorImplEJN3c1010TensorImpl8ImplTypeENS3_7StorageENS3_14DispatchKeySetEN6caffe28TypeMetaERNS3_13intrusive_ptrINS_9QuantizerENS3_6detail34intrusive_target_default_null_typeISB_EEEEEEENS_6TensorEDpOT0_":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6detail11make_tensorINS_11QTensorImplEJN3c1010TensorImpl8ImplTypeENS3_7StorageENS3_14DispatchKeySetEN6caffe28TypeMetaERNS3_13intrusive_ptrINS_9QuantizerENS3_6detail34intrusive_target_default_null_typeISB_EEEEEEENS_6TensorEDpOT0_")

        if fn_name == "_ZN3c107StorageC2ERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZN3c107StorageC2ERKS0_")

        if fn_name == "_ZNK2at10TensorBase7key_setEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at10TensorBase7key_setEv")

        if fn_name == "_ZN3c1013intrusive_ptrIN2at9QuantizerENS_6detail34intrusive_target_default_null_typeIS2_EEEC2EOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1013intrusive_ptrIN2at9QuantizerENS_6detail34intrusive_target_default_null_typeIS2_EEEC2EOS6_")

        if fn_name == "_ZN2at6nativeL26create_subtensor_quantizerERKNS_6TensorEbllll":
            raise NotImplementedError(f"Unhandled call: _ZN2at6nativeL26create_subtensor_quantizerERKNS_6TensorEbllll")

        if fn_name == "_ZN2at19_nested_from_paddedERKNS_6TensorES2_b":
            raise NotImplementedError(f"Unhandled call: _ZN2at19_nested_from_paddedERKNS_6TensorES2_b")

        if fn_name == "_ZN2at6native6bmm_nnERNS_6TensorERKS1_S4_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native6bmm_nnERNS_6TensorERKS1_S4_")

        if fn_name == "_ZSt3getILm0EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm0EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_")

        if fn_name == "_ZN2at27_transform_bias_rescale_qkvERKNS_6TensorES2_l":
            raise NotImplementedError(f"Unhandled call: _ZN2at27_transform_bias_rescale_qkvERKNS_6TensorES2_l")

        if fn_name == "_ZN2at6native14masked_softmaxERNS_6TensorESt8optionalIS1_ERKS1_S3_IlE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native14masked_softmaxERNS_6TensorESt8optionalIS1_ERKS1_S3_IlE")

        if fn_name == "_ZN2at6native6bmm_ntERKNS_6TensorES3_":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native6bmm_ntERKNS_6TensorES3_")

        if fn_name == "_ZSt3getILm1EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm1EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_")

        if fn_name == "_ZSt3getILm2EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_":
            raise NotImplementedError(
                f"Unhandled call: _ZSt3getILm2EJN2at6TensorES1_S1_EEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS6_")

        if fn_name == "_ZN2at6native17svd_uses_cusolverERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native17svd_uses_cusolverERKNS_6TensorE")

        if fn_name == "_ZN2at14expand_inplaceERKNS_6TensorES2_PKc":
            raise NotImplementedError(f"Unhandled call: _ZN2at14expand_inplaceERKNS_6TensorES2_PKc")

        if fn_name == "_ZN2at20TensorIteratorConfig24enforce_linear_iterationEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at20TensorIteratorConfig24enforce_linear_iterationEb")

        if fn_name == "_ZN2at6native7cpublas10could_packEN3c1010ScalarTypeE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native7cpublas10could_packEN3c1010ScalarTypeE")

        if fn_name == "_ZN2at6native21should_use_acc_bufferERNS_14TensorIteratorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native21should_use_acc_bufferERNS_14TensorIteratorE")

        if fn_name == "_ZNK2at6Tensor4sortElb":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor4sortElb")

        if fn_name == "_ZN2at6native23create_reduction_resultERKNS_6TensorEN3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native23create_reduction_resultERKNS_6TensorEN3c1016OptionalArrayRefIlEEbNS4_10ScalarTypeE")

        if fn_name == "_ZN2at6native19get_dtype_from_selfERKNS_6TensorERKSt8optionalIN3c1010ScalarTypeEEb":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native19get_dtype_from_selfERKNS_6TensorERKSt8optionalIN3c1010ScalarTypeEEb")

        if fn_name == "_ZN2at6native12_GLOBAL__N_118quantized_cat_implILb0EEENS_6TensorEN3c108IListRefIS3_EEldl":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_118quantized_cat_implILb0EEENS_6TensorEN3c108IListRefIS3_EEldl")

        if fn_name == "_ZNK2at6Tensor8to_denseESt8optionalIN3c1010ScalarTypeEES1_IbE":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8to_denseESt8optionalIN3c1010ScalarTypeEES1_IbE")

        if fn_name == "_ZN2at32_sparse_compressed_tensor_unsafeERKNS_6TensorES2_S2_N3c108ArrayRefIlEENS3_13TensorOptionsE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at32_sparse_compressed_tensor_unsafeERKNS_6TensorES2_S2_N3c108ArrayRefIlEENS3_13TensorOptionsE")

        if fn_name == "_ZNK2at6Tensor9expand_asERKS0_":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor9expand_asERKS0_")

        if fn_name == "_ZN2at4onesEN3c108ArrayRefIlEENS0_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at4onesEN3c108ArrayRefIlEENS0_13TensorOptionsE")

        if fn_name == "_ZNSt8optionalIN3c106LayoutEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8optionalIN3c106LayoutEEC2IS1_TnNSt9enable_ifIX7__and_vISt6__not_ISt7is_sameIS2_NSt9remove_cvINSt16remove_referenceIT_E4typeEE4typeEEES5_IS6_ISt10in_place_tSD_EESt16is_constructibleIS1_JS9_EESt14is_convertibleIS9_S1_EEEbE4typeELb1EEEOS9_")

        if fn_name == "llvm.fshl.i64":
            raise NotImplementedError(f"Unhandled call: llvm.fshl.i64")

        if fn_name == "llvm.fshl.i32":
            raise NotImplementedError(f"Unhandled call: llvm.fshl.i32")

        if fn_name == "_ZNK2at6native16NestedTensorImpl4sizeEl":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6native16NestedTensorImpl4sizeEl")

        if fn_name == "_ZNSt6vectorIN3c108ArrayRefIlEESaIS2_EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNSt6vectorIN3c108ArrayRefIlEESaIS2_EEixEm")

        if fn_name == "_ZN2at6native22NestedTensor_get_sizesEPKNS0_16NestedTensorImplE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native22NestedTensor_get_sizesEPKNS0_16NestedTensorImplE")

        if fn_name == "_ZN2at4impl31variable_excluded_from_dispatchEv":
            raise NotImplementedError(f"Unhandled call: _ZN2at4impl31variable_excluded_from_dispatchEv")

        if fn_name == "_ZNOSt8optionalIN3c1012MemoryFormatEE8value_orIS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNOSt8optionalIN3c1012MemoryFormatEE8value_orIS1_EES1_OT_")

        if fn_name == "_ZNK3c1013TensorOptions17memory_format_optEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c1013TensorOptions17memory_format_optEv")

        if fn_name == "_ZN2at6nativeL23verify_empty_parametersERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL23verify_empty_parametersERKNS_6TensorESt8optionalIN3c1010ScalarTypeEES4_INS5_6LayoutEES4_INS5_6DeviceEES4_IbES4_INS5_12MemoryFormatEE")

        if fn_name == "_ZN3c10neIlEEbRKSt6vectorIT_SaIS2_EENS_8ArrayRefIS2_EE":
            raise NotImplementedError(f"Unhandled call: _ZN3c10neIlEEbRKSt6vectorIT_SaIS2_EENS_8ArrayRefIS2_EE")

        if fn_name == "_ZN3c1017multiply_integersISt6vectorIlSaIlEETnNSt9enable_ifIXsr3stdE13is_integral_vINT_10value_typeEEEiE4typeELi0EEElRKS5_":
            raise NotImplementedError(
                f"Unhandled call: _ZN3c1017multiply_integersISt6vectorIlSaIlEETnNSt9enable_ifIXsr3stdE13is_integral_vINT_10value_typeEEEiE4typeELi0EEElRKS5_")

        if fn_name == "_ZNKRSt8optionalIN3c106LayoutEE8value_orIS1_EES1_OT_":
            raise NotImplementedError(f"Unhandled call: _ZNKRSt8optionalIN3c106LayoutEE8value_orIS1_EES1_OT_")

        if fn_name == "_ZN2at6native12_GLOBAL__N_126all_inputs_sharing_qparamsERKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS6_EE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_126all_inputs_sharing_qparamsERKSt6vectorISt17reference_wrapperIKNS_6TensorEESaIS6_EE")

        if fn_name == "_ZNK3c108ArrayRefIN2at6TensorEE5beginEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c108ArrayRefIN2at6TensorEE5beginEv")

        if fn_name == "_ZNKSt8optionalIbE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIbE9has_valueEv")

        if fn_name == "_ZN2at6native12_fft_c2r_mklERKNS_6TensorEN3c108ArrayRefIlEEll":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native12_fft_c2r_mklERKNS_6TensorEN3c108ArrayRefIlEEll")

        if fn_name == "_ZNKSt8optionalIN3c106ScalarEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c106ScalarEE9has_valueEv")

        if fn_name == "_ZNK2at6Tensor9new_zerosEN3c108ArrayRefIlEENS1_13TensorOptionsE":
            raise NotImplementedError(
                f"Unhandled call: _ZNK2at6Tensor9new_zerosEN3c108ArrayRefIlEENS1_13TensorOptionsE")

        if fn_name == "_ZNK2at6Tensor8isfiniteEv":
            raise NotImplementedError(f"Unhandled call: _ZNK2at6Tensor8isfiniteEv")

        if fn_name == "_ZN2at4_ops17normal_functional4callERKNS_6TensorEddSt8optionalINS_9GeneratorEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at4_ops17normal_functional4callERKNS_6TensorEddSt8optionalINS_9GeneratorEE")

        if fn_name == "_ZNSt8optionalIN2at9GeneratorEEC2ERKS2_":
            raise NotImplementedError(f"Unhandled call: _ZNSt8optionalIN2at9GeneratorEEC2ERKS2_")

        if fn_name == "_ZNKSt6bitsetILm64EE9referencecvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt6bitsetILm64EE9referencecvbEv")

        if fn_name == "_ZNSt6bitsetILm64EEixEm":
            raise NotImplementedError(f"Unhandled call: _ZNSt6bitsetILm64EEixEm")

        if fn_name == "_ZNKR2at17OptionalTensorRef12getTensorRefEv":
            raise NotImplementedError(f"Unhandled call: _ZNKR2at17OptionalTensorRef12getTensorRefEv")

        if fn_name == "_ZN9__gnu_cxxneIPSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_":
            raise NotImplementedError(
                f"Unhandled call: _ZN9__gnu_cxxneIPSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_")

        if fn_name == "_ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5beginEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE5beginEv")

        if fn_name == "_ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE3endEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt6vectorISt17reference_wrapperIKN2at6TensorEESaIS4_EE3endEv")

        if fn_name == "_ZN2at6native22cat_should_skip_tensorERKNS_6TensorE":
            raise NotImplementedError(f"Unhandled call: _ZN2at6native22cat_should_skip_tensorERKNS_6TensorE")

        if fn_name == "_ZNK9__gnu_cxx17__normal_iteratorIPSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEdeEv":
            raise NotImplementedError(
                f"Unhandled call: _ZNK9__gnu_cxx17__normal_iteratorIPSt17reference_wrapperIKN2at6TensorEESt6vectorIS5_SaIS5_EEEdeEv")

        if fn_name == "_ZN2at6nativeL23make_index_put_iteratorERKNS0_13AdvancedIndexERKNS_6TensorE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6nativeL23make_index_put_iteratorERKNS0_13AdvancedIndexERKNS_6TensorE")

        if fn_name == "_ZN2at6native12_GLOBAL__N_140_make_unfold_backward_iter_over_grad_outERNS_6TensorERKS2_lll":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native12_GLOBAL__N_140_make_unfold_backward_iter_over_grad_outERNS_6TensorERKS2_lll")

        if fn_name == "_ZNKSt8optionalIN3c1010ScalarTypeEEcvbEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN3c1010ScalarTypeEEcvbEv")

        if fn_name == "_ZNSt8functionIFbRKlEEC2IZN2at6native12_GLOBAL__N_122check_maxpool2d_paramsEN3c108ArrayRefIlEESA_SA_SA_E3$_0vEEOT_":
            raise NotImplementedError(
                f"Unhandled call: _ZNSt8functionIFbRKlEEC2IZN2at6native12_GLOBAL__N_122check_maxpool2d_paramsEN3c108ArrayRefIlEESA_SA_SA_E3$_0vEEOT_")

        if fn_name == "_ZNK3c106Scalar15isFloatingPointEv":
            raise NotImplementedError(f"Unhandled call: _ZNK3c106Scalar15isFloatingPointEv")

        if fn_name == "_ZN2at6native7DEFAULT17is_reduce_lastdimERNS_18TensorIteratorBaseE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at6native7DEFAULT17is_reduce_lastdimERNS_18TensorIteratorBaseE")

        if fn_name == "_ZNKR3c1016OptionalArrayRefIlE8value_orINS_8ArrayRefIlEEEENSt9enable_ifIXsr3stdE16is_convertible_vIOT_S4_EES4_E4typeES7_":
            raise NotImplementedError(
                f"Unhandled call: _ZNKR3c1016OptionalArrayRefIlE8value_orINS_8ArrayRefIlEEEENSt9enable_ifIXsr3stdE16is_convertible_vIOT_S4_EES4_E4typeES7_")

        if fn_name == "_ZNKSt8optionalIN2at9GeneratorEE9has_valueEv":
            raise NotImplementedError(f"Unhandled call: _ZNKSt8optionalIN2at9GeneratorEE9has_valueEv")

        if fn_name == "_ZN2at13scalar_tensorERKN3c106ScalarENS0_13TensorOptionsE":
            raise NotImplementedError(f"Unhandled call: _ZN2at13scalar_tensorERKN3c106ScalarENS0_13TensorOptionsE")

        if fn_name == "_ZN2at15empty_quantizedEN3c108ArrayRefIlEERKNS_6TensorENS0_13TensorOptionsESt8optionalINS0_12MemoryFormatEE":
            raise NotImplementedError(
                f"Unhandled call: _ZN2at15empty_quantizedEN3c108ArrayRefIlEERKNS_6TensorENS0_13TensorOptionsESt8optionalINS0_12MemoryFormatEE")

        if fn_name == "_ZN2at8choleskyERKNS_6TensorEb":
            raise NotImplementedError(f"Unhandled call: _ZN2at8choleskyERKNS_6TensorEb")

        raise NotImplementedError(f"Unhandled call: {fn_name}")

    # Recursive expression builder
    def build_expr(self, node):
        inst = node.get("inst")
        ops = node.get("ops", [])

        if inst == "const_int":
            return IntVal(int(node["const"]))

        if inst == "const_fp":
            return RealVal(float(node["const"]))

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
        if inst == "fcmp":
            pred = int(ops[0]["const"])
            lhs, rhs = self.build_expr(ops[1]), self.build_expr(ops[2])
            return FCmpMapper.apply(pred, lhs, rhs)
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

        if inst == "and":
            return self.build_expr(ops[0]) & self.build_expr(ops[1])
        if inst == "or":
            return self.build_expr(ops[0]) | self.build_expr(ops[1])
        if inst == "xor":
            return self.build_expr(ops[0]) ^ self.build_expr(ops[1])

        raise NotImplementedError(f"Unhandled instruction: {inst}")


# ----------------------------
# Solver builder
# ----------------------------
class SolverBuilder:
    def __init__(self, model: TensorModel, mapper: FunctionMapper):
        self.model = model
        self.mapper = mapper

    def build_solver_from_data(self, path, solver):
        for step in reversed(path.get("detail", [])):
            cond = step.get("condition")
            if cond is None:
                continue
            # if not self.contains_const_arg(cond):
            #     continue
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
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("extracted_smt/_ZN2at6native5logitERKNS_6TensorESt8optionalIdE.json")
