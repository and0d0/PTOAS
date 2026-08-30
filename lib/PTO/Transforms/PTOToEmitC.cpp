// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic ignored "-Woverloaded-virtual"
// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8

#include <cassert>
#include <climits>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/MemoryConsistencyAttrs.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                   
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

static std::string getElemTypeStringForGT(Type elemTy);
static bool getStaticMemrefLayout(MemRefType mrTy,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs);
static void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D);
static std::string joinIntTemplateParams(ArrayRef<int64_t> values);
static SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape);
static std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum = "pto::Layout::ND");
static emitc::OpaqueType getRuntimeGlobalTensorOpaqueType(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum = "pto::Layout::ND");

static const char *addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

static pto::AddressSpace getAddressSpaceOrGM(Attribute memorySpace) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    return asAttr.getAddressSpace();
  }
  return pto::AddressSpace::GM;
}

static Type getEmitCVariableResultType(Type valueType) {
  return valueType;
}

static Value loadEmitCVariableIfNeeded(OpBuilder &builder, Location loc,
                                       Value value) {
  (void)builder;
  (void)loc;
  return value;
}

static Value getSourceEmitCVariable(Value value) {
  if (value.getDefiningOp<emitc::VariableOp>()) {
    return value;
  }
  return {};
}

static void appendRawLocationNameHints(Location loc,
                                       SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string raw = nameLoc.getName().getValue().str();
    if (!raw.empty()) {
      hints.push_back(std::move(raw));
    }
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        std::string raw = strAttr.getValue().str();
        if (!raw.empty()) {
          hints.push_back(std::move(raw));
        }
        return;
      }
      if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
        for (Attribute attr : arrayAttr) {
          auto strAttr = dyn_cast<StringAttr>(attr);
          if (!strAttr) {
            continue;
          }
          std::string raw = strAttr.getValue().str();
          if (!raw.empty()) {
            hints.push_back(std::move(raw));
          }
        }
        if (!hints.empty()) {
          return;
        }
      }
    }

    // Only metadata explicitly attached by PTOAS name-hint recovery carries an
    // ordered result-name list. Ordinary fused child locations are debug
    // provenance, not result-indexed name hints.
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendRawLocationNameHints(callSiteLoc.getCallee(), hints);
    if (hints.empty()) {
      appendRawLocationNameHints(callSiteLoc.getCaller(), hints);
    }
  }
}

static Location getIndexedNameHintLoc(Location fallbackLoc, unsigned index) {
  SmallVector<std::string, 4> hints;
  appendRawLocationNameHints(fallbackLoc, hints);
  if (index >= hints.size() || hints[index].empty()) {
    return fallbackLoc;
  }
  return NameLoc::get(StringAttr::get(fallbackLoc.getContext(), hints[index]),
                      fallbackLoc);
}

static constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";
static constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
static constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
static constexpr llvm::StringLiteral kPipePeerDirMaskAttrName =
    "__pto.peer_dir_mask";
static constexpr llvm::StringLiteral kEmitCScalarOutTypeAttrName =
    "__pto.emitc_scalar_out_type";
static constexpr llvm::StringLiteral kLastUseAttrName = "pto.last_use";
static constexpr llvm::StringLiteral kLastUseMarkerPrefix = "PTOAS__LAST_USE__";

static int64_t getAPIntSignedValue(const APInt &value) {
  return value.getBitWidth() == 0 ? 0 : value.getSExtValue();
}

static uint64_t getAPIntUnsignedValue(const APInt &value) {
  return value.getBitWidth() == 0 ? 0 : value.getZExtValue();
}

static int64_t getIntegerAttrSignedValue(IntegerAttr attr) {
  return getAPIntSignedValue(attr.getValue());
}

static SmallVector<unsigned, 4> collectTileOperandNumbers(Operation *op) {
  SmallVector<unsigned, 4> tileOperandNumbers;
  for (OpOperand &operand : op->getOpOperands()) {
    if (isa<pto::TileBufType>(operand.get().getType())) {
      tileOperandNumbers.push_back(operand.getOperandNumber());
    }
  }
  return tileOperandNumbers;
}

static bool isDpsInitOperand(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    for (OpOperand &init : dpsIface.getDpsInitsMutable()) {
      if (&init == &operand) {
        return true;
      }
    }
  }
  return false;
}

static SmallVector<unsigned, 4>
buildDefaultLastUseTileSlotOrder(Operation *op) {
  SmallVector<unsigned, 4> dpsInitTileOperands;
  SmallVector<unsigned, 4> nonDpsTileOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<pto::TileBufType>(operand.get().getType())) {
      continue;
    }
    if (isDpsInitOperand(operand)) {
      dpsInitTileOperands.push_back(operand.getOperandNumber());
    } else {
      nonDpsTileOperands.push_back(operand.getOperandNumber());
    }
  }

  // Most tile intrinsics lower as `CALLEE(dst, src0, src1, ...)`. When an op
  // has exactly one DPS init tile, treat that output slot as the leading
  // emitted tile operand so `[[pto::last_use(...)]]` aligns with the final
  // intrinsic call order.
  if (dpsInitTileOperands.size() == 1) {
    SmallVector<unsigned, 4> ordered{dpsInitTileOperands.front()};
    ordered.append(nonDpsTileOperands.begin(), nonDpsTileOperands.end());
    return ordered;
  }

  SmallVector<unsigned, 4> ordered = std::move(nonDpsTileOperands);
  ordered.append(dpsInitTileOperands.begin(), dpsInitTileOperands.end());
  return ordered;
}

static std::optional<std::string> buildLastUseMarkerCallee(Operation *op,
                                                           StringRef callee,
                                                           ArrayRef<unsigned> tileSlotOrder = {}) {
  auto lastUseAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      op->getAttr(kLastUseAttrName));
  if (!lastUseAttr) {
    return std::nullopt;
  }

  SmallVector<unsigned, 4> originalTileOperands = collectTileOperandNumbers(op);
  ArrayRef<int64_t> originalBits = lastUseAttr.asArrayRef();
  if (originalTileOperands.size() != originalBits.size()) {
    return std::nullopt;
  }

  SmallVector<unsigned, 4> defaultTileSlotOrder;
  if (tileSlotOrder.empty()) {
    defaultTileSlotOrder = buildDefaultLastUseTileSlotOrder(op);
    tileSlotOrder = defaultTileSlotOrder;
  }
  if (tileSlotOrder.size() != originalBits.size()) {
    return std::nullopt;
  }

  SmallVector<int64_t, 4> reorderedBits;
  reorderedBits.reserve(tileSlotOrder.size());
  for (unsigned operandNumber : tileSlotOrder) {
    bool found = false;
    for (auto [idx, originalOperandNumber] : llvm::enumerate(originalTileOperands)) {
      if (originalOperandNumber != operandNumber) {
        continue;
      }
      reorderedBits.push_back(originalBits[idx]);
      found = true;
      break;
    }
    if (!found) {
      return std::nullopt;
    }
  }

  std::string marker = kLastUseMarkerPrefix.str();
  marker.append(callee.str());
  marker.append("__");
  bool first = true;
  for (int64_t bit : reorderedBits) {
    if (!first) {
      marker.append("__");
    }
    first = false;
    marker.append(std::to_string(bit));
  }
  return marker;
}

static StringRef getLastUseAwareCallee(Operation *op, StringRef callee,
                                       std::string &storage,
                                       ArrayRef<unsigned> tileSlotOrder = {}) {
  std::optional<std::string> marker =
      buildLastUseMarkerCallee(op, callee, tileSlotOrder);
  if (!marker) {
    return callee;
  }
  storage = std::move(*marker);
  return storage;
}

static void createOpaqueCall(ConversionPatternRewriter &rewriter, Location loc,
                             TypeRange resultTypes, StringRef callee,
                             ArrayAttr args, ArrayAttr templateArgs,
                             ValueRange operands) {
  rewriter.create<emitc::CallOpaqueOp>(loc, resultTypes, callee, args,
                                       templateArgs, operands);
}

static void createLastUseAwareOpaqueCall(
    ConversionPatternRewriter &rewriter, Operation *op, TypeRange resultTypes,
    StringRef callee, ValueRange operands, ArrayAttr args = ArrayAttr{},
    ArrayAttr templateArgs = ArrayAttr{},
    ArrayRef<unsigned> tileSlotOrder = {}) {
  std::string calleeStorage;
  StringRef effectiveCallee =
      getLastUseAwareCallee(op, callee, calleeStorage, tileSlotOrder);
  createOpaqueCall(rewriter, op->getLoc(), resultTypes, effectiveCallee, args,
                   templateArgs, operands);
}

static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy, Operation *anchor,
                                         StringRef tag = {});

static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor, StringRef tag = {});

static std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout")) {
    return attr.getLayout();
  }
  return std::nullopt;
}

static std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (Operation *def = v.getDefiningOp()) {
    if (auto layout = getLayoutAttrFromOp(def)) {
      return layout;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(subview.getSource());
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(reinterpret.getSource());
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(cast.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0) {
        break;
      }
      v = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    break;
  }
  return std::nullopt;
}

static std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor)) {
    return layout;
  }
  return resolveLayoutFromValueChain(basePtr);
}

static std::string layoutToEmitCString(mlir::pto::Layout layout) {
  switch (layout) {
  case mlir::pto::Layout::ND:
    return "pto::Layout::ND";
  case mlir::pto::Layout::DN:
    return "pto::Layout::DN";
  case mlir::pto::Layout::NZ:
    return "pto::Layout::NZ";
  case mlir::pto::Layout::MX_A_ZZ:
    return "pto::Layout::MX_A_ZZ";
  case mlir::pto::Layout::MX_B_NN:
    return "pto::Layout::MX_B_NN";
  }
  return "pto::Layout::ND";
}

static bool isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

static Value peelGlobalTensorConversionBridge(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
    return value;
  }

  Value input = cast.getOperand(0);
  if (isEmitCGlobalTensorLikeType(input.getType()) &&
      isEmitCGlobalTensorLikeType(value.getType())) {
    return input;
  }
  return value;
}

static bool isF8E8M0ElemType(Type elemTy) {
  return mlir::pto::isPTOF8E8M0Type(elemTy);
}

static std::string getEmitCScalarTypeToken(Type elemTy) {
  if (pto::isPTOFloat8E4M3LikeType(elemTy)) {
    return "float8_e4m3_t";
  }
  if (pto::isPTOFloat8E5M2LikeType(elemTy)) {
    return "float8_e5m2_t";
  }
  if (isF8E8M0ElemType(elemTy)) {
    return "float8_e8m0_t";
  }
  if (isa<pto::HiF8Type>(elemTy)) {
    return "hifloat8_t";
  }
  if (isa<pto::F4E1M2x2Type>(elemTy)) {
    return "float4_e1m2x2_t";
  }
  if (isa<pto::F4E2M1x2Type>(elemTy)) {
    return "float4_e2m1x2_t";
  }
  if (elemTy.isF16()) {
    return "half";
  }
  if (elemTy.isBF16()) {
    return "bfloat16_t";
  }
  if (elemTy.isF32()) {
    return "float";
  }
  if (elemTy.isF64()) {
    return "double";
  }
  if (elemTy.isInteger(8)) {
    return (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8)) ? "int8_t"
                                                                       : "uint8_t";
  }
  if (elemTy.isInteger(16)) {
    return (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
               ? "int16_t"
               : "uint16_t";
  }
  if (elemTy.isInteger(32)) {
    return (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
               ? "int32_t"
               : "uint32_t";
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  }
  return "float";
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef pointeeTypeStr) {
  return emitc::PointerType::get(emitc::OpaqueType::get(ctx, pointeeTypeStr));
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef qualifier,
                                              StringRef elemTypeStr) {
  return getEmitCPointerType(ctx, (qualifier + " " + elemTypeStr).str());
}

static bool isEmitCPointerLikeType(Type ty) {
  if (isa<emitc::PointerType>(ty)) {
    return true;
  }
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty)) {
    return opaqueTy.getValue().ends_with("*");
  }
  return false;
}

static int64_t getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == 1) {
    return 1;
  }
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(16)) {
    return 2;
  }
  if (elemTy.isF32() || elemTy.isInteger(32)) {
    return 4;
  }
  if (elemTy.isF64() || elemTy.isInteger(64)) {
    return 8;
  }
  return 4;
}

// ---------------------------------------------------------------------------
// !pto.struct support: a deterministic C++ type name + file-scope definition.
// ---------------------------------------------------------------------------

// Replace any character that is not a C++ identifier character with '_'. The
// scalar tokens below are already identifier-safe; this is defensive.
static std::string sanitizeIdentifier(std::string s) {
  for (char &c : s) {
    bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') || c == '_';
    if (!ok) {
      c = '_';
    }
  }
  return s;
}

// Mangle a scalar-storable field type into a C++-identifier-safe token. The
// encoding is injective, so distinct struct types never collide on a name:
//   - scalar:        the MLIR type spelling (f16, bf16, i8, si32, ui32, ...)
//   - nested struct: S_<f0>_<f1>_..._E  (S/E delimiters disambiguate nesting)
//
// Scalars are mangled from the MLIR spelling rather than from
// getEmitCScalarTypeToken(): that token is many-to-one (i32 and si32 both give
// "int32_t"), which would emit two `struct` definitions under one name and
// break the generated C++ with a redefinition. MLIR type printing is injective,
// and the struct verifier restricts fields to types whose spellings are already
// pure identifier characters, so this mangling is collision-free by
// construction.
static std::string mangleStructFieldType(Type t) {
  if (auto st = dyn_cast<pto::StructType>(t)) {
    std::string s = "S";
    for (Type f : st.getFieldTypes()) {
      s += "_" + mangleStructFieldType(f);
    }
    return s + "_E";
  }
  std::string spelling;
  llvm::raw_string_ostream os(spelling);
  t.print(os);
  return sanitizeIdentifier(os.str());
}

// Stable, content-derived C++ type name for a !pto.struct, e.g.
// !pto.struct<f16, i8> -> "PtoStruct_f16_i8". A pure function of the type, so
// the type converter and the file-scope definition emitter agree without any
// shared state.
static std::string getStructTypeName(pto::StructType st) {
  std::string s = "PtoStruct";
  for (Type f : st.getFieldTypes()) {
    s += "_" + mangleStructFieldType(f);
  }
  return s;
}

// Render a single struct field declaration `<cppType> <name>;`.
static std::string renderStructFieldDecl(Type fieldTy,
                                         const std::string &name) {
  if (auto st = dyn_cast<pto::StructType>(fieldTy)) {
    return getStructTypeName(st) + " " + name + ";";
  }
  return getEmitCScalarTypeToken(fieldTy) + " " + name + ";";
}

// Render the full C++ definition of a !pto.struct as file-scope text.
static std::string renderStructDef(pto::StructType st) {
  std::string s = "struct " + getStructTypeName(st) + " {\n";
  for (auto [i, f] : llvm::enumerate(st.getFieldTypes())) {
    s += "  " + renderStructFieldDecl(f, "f" + std::to_string(i)) + "\n";
  }
  return s + "};";
}

// Collect every !pto.struct reachable from `t` into `out` in definition order:
// a nested struct is inserted before the struct that embeds it, so emitting in
// `out` order produces valid C++ (no use-before-definition).
static void collectStructTypes(Type t, llvm::SetVector<pto::StructType> &out) {
  auto st = dyn_cast<pto::StructType>(t);
  if (!st || out.contains(st)) {
    return;
  }
  for (Type f : st.getFieldTypes()) {
    collectStructTypes(f, out);
  }
  out.insert(st);
}

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr);
static pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr);
static pto::SLayout getTileBufSLayoutValue(pto::TileBufConfigAttr configAttr);
static int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx);
static bool isLowPrecisionCubeOperandType(Type elemTy) {
  return pto::isPTOFloat8Type(elemTy) || isa<pto::F4E1M2x2Type>(elemTy) ||
         isa<pto::F4E2M1x2Type>(elemTy);
}

struct SpecialGlobalTensorTypeSpec {
  std::string shapeTypeExpr;
  std::string strideTypeExpr;
  std::string layoutEnum;
};

static std::optional<SpecialGlobalTensorTypeSpec>
getSpecialScaleGlobalTensorTypeSpecForTileValue(Value dstValue,
                                                ArrayRef<int64_t> shape,
                                                Type elemTy) {
  dstValue = peelUnrealized(dstValue);

  auto dstTileTy = dyn_cast<pto::TileBufType>(dstValue.getType());
  if (!dstTileTy) {
    return std::nullopt;
  }

  auto dstSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(
      dstTileTy.getMemorySpace());
  if (!dstSpace || dstSpace.getAddressSpace() != pto::AddressSpace::MAT) {
    return std::nullopt;
  }

  ArrayRef<int64_t> effectiveShape = dstTileTy.getShape();
  if (effectiveShape.empty()) {
    effectiveShape = shape;
  }
  auto config = dstTileTy.getConfigAttr();
  if (!isF8E8M0ElemType(elemTy)) {
    return std::nullopt;
  }
  if (effectiveShape.size() != 2) {
    return std::nullopt;
  }

  pto::BLayout blayout = getTileBufBLayoutValue(config);
  pto::SLayout slayout = getTileBufSLayoutValue(config);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);

  if (blayout == pto::BLayout::RowMajor &&
      slayout == pto::SLayout::RowMajor) {
    if (effectiveShape[0] == 1) {
      return std::nullopt;
    }
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_A_ZZ>",
        "BaseShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_A_ZZ>",
        "pto::Layout::MX_A_ZZ",
    };
  }

  if (blayout == pto::BLayout::ColMajor &&
      slayout == pto::SLayout::ColMajor) {
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_B_NN>",
        "BaseShape2D<" + elemTypeStr + ", " +
            std::to_string(effectiveShape[0]) + ", " +
            std::to_string(effectiveShape[1]) + ", pto::Layout::MX_B_NN>",
        "pto::Layout::MX_B_NN",
    };
  }

  return std::nullopt;
}

static std::optional<SpecialGlobalTensorTypeSpec>
getSpecialGlobalTensorTypeSpecForLayout(std::optional<mlir::pto::Layout> layout,
                                        ArrayRef<int64_t> shape, Type elemTy) {
  if (!layout || !isF8E8M0ElemType(elemTy) || shape.size() != 2) {
    return std::nullopt;
  }

  auto alignUp = [](int64_t value, int64_t align) -> int64_t {
    if (value < 0 || align <= 0) {
      return value;
    }
    return ((value + align - 1) / align) * align;
  };

  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  switch (*layout) {
  case mlir::pto::Layout::MX_A_ZZ: {
    int64_t rows = alignUp(shape[0], 16);
    int64_t cols = alignUp(shape[1], 2);
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_A_ZZ>",
        "BaseShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_A_ZZ>",
        "pto::Layout::MX_A_ZZ",
    };
  }
  case mlir::pto::Layout::MX_B_NN: {
    int64_t rows = alignUp(shape[0], 2);
    int64_t cols = alignUp(shape[1], 16);
    return SpecialGlobalTensorTypeSpec{
        "TileShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_B_NN>",
        "BaseShape2D<" + elemTypeStr + ", " + std::to_string(rows) + ", " +
            std::to_string(cols) + ", pto::Layout::MX_B_NN>",
        "pto::Layout::MX_B_NN",
    };
  }
  default:
    return std::nullopt;
  }
}

static std::optional<SpecialGlobalTensorTypeSpec>
getSpecialScaleGlobalTensorTypeSpec(Operation *anchor, MemRefType mrTy) {
  auto load = dyn_cast_or_null<pto::TLoadOp>(anchor);
  if (!load) {
    return std::nullopt;
  }
  return getSpecialScaleGlobalTensorTypeSpecForTileValue(
      load.getDst(), mrTy.getShape(), mrTy.getElementType());
}

static const char *scalingRoleToken(Type elemTy,
                                    pto::TileBufConfigAttr configAttr) {
  if (!isF8E8M0ElemType(elemTy)) {
    return "TileType::Scaling";
  }
  pto::BLayout bl = getTileBufBLayoutValue(configAttr);
  pto::SLayout sl = getTileBufSLayoutValue(configAttr);
  if (bl == pto::BLayout::RowMajor && sl == pto::SLayout::RowMajor) {
    return "TileType::ScaleLeft";
  }
  if (bl == pto::BLayout::ColMajor && sl == pto::SLayout::ColMajor) {
    return "TileType::ScaleRight";
  }
  return "TileType::Scaling";
}

static const char *tileRoleToken(Attribute memorySpace,
                                 std::optional<Type> elemType = std::nullopt,
                                 std::optional<pto::TileBufConfigAttr> configAttr = std::nullopt) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    switch (asAttr.getAddressSpace()) {
    case pto::AddressSpace::VEC:
      return "TileType::Vec";
    case pto::AddressSpace::MAT:
      return "TileType::Mat";
    case pto::AddressSpace::LEFT:
      return "TileType::Left";
    case pto::AddressSpace::RIGHT:
      return "TileType::Right";
    case pto::AddressSpace::ACC:
      return "TileType::Acc";
    case pto::AddressSpace::BIAS:
      return "TileType::Bias";
    case pto::AddressSpace::SCALING:
      if (elemType && configAttr) {
        return scalingRoleToken(*elemType, *configAttr);
      }
      return "TileType::Scaling";
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return "TileType::Vec";
    }
  }
  return "TileType::Vec";
}

static const char *inferScalingRoleFromValue(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy) {
    return nullptr;
  }
  StringRef token = opaqueTy.getValue();
  if (token.contains("TileType::ScaleLeft")) {
    return "TileType::ScaleLeft";
  }
  if (token.contains("TileType::ScaleRight")) {
    return "TileType::ScaleRight";
  }
  if (token.contains("TileType::Scaling")) {
    return "TileType::Scaling";
  }
  return nullptr;
}

static std::string tileBufCompactToken(pto::TileBufConfigAttr configAttr) {
  std::string compactTok = "CompactMode::Null";
  if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
    switch (static_cast<int32_t>(compactAttr.getValue())) {
    case 1:
      compactTok = "CompactMode::Normal";
      break;
    case 2:
      compactTok = "CompactMode::RowPlusOne";
      break;
    default:
      compactTok = "CompactMode::Null";
      break;
    }
  }
  return compactTok;
}

static std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type) {
  if (type.getRank() != 2) {
    return std::nullopt;
  }
  auto validShape = type.getValidShape();
  if (validShape.size() != 2) {
    return std::nullopt;
  }

  Type elemTy = type.getElementType();
  auto configAttr = type.getConfigAttr();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  auto render = [elemTy, blayout](int64_t dim, int dimIdx) {
    return renderTileTemplateDim(dim, elemTy, blayout, dimIdx);
  };

  std::string vrowTok =
      validShape[0] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[0], 0));
  std::string vcolTok =
      validShape[1] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[1], 1));

  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace())) {
    if (isLowPrecisionCubeOperandType(elemTy)) {
      if (asAttr.getAddressSpace() == pto::AddressSpace::LEFT &&
          shape[0] != 1 &&
          validShape[1] != ShapedType::kDynamic) {
        vcolTok = std::to_string(render(cols, 1));
      } else if (asAttr.getAddressSpace() == pto::AddressSpace::RIGHT &&
                 validShape[0] != ShapedType::kDynamic) {
        vrowTok = std::to_string(render(rows, 0));
      }
    }
  }

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize())) {
    fractal = static_cast<int32_t>(getIntegerAttrSignedValue(frAttr));
  }

  return std::string("Tile<") +
         tileRoleToken(type.getMemorySpace(), elemTy, type.getConfigAttr()) + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " +
         std::to_string(render(rows, 0)) + ", " +
         std::to_string(render(cols, 1)) + ", " +
         tileBufBLayoutToken(configAttr) + ", " + vrowTok + ", " + vcolTok +
         ", " + tileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ", " +
         tileBufCompactToken(configAttr) + ">";
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch targetArch) {
    // ---------------------------------------------------------
    // 1. 基本类型 (f32, i32, index)
    // ---------------------------------------------------------
    addConversion([Ctx](FloatType type) -> Type {
      if (pto::isPTOFloat8E4M3LikeType(type)) {
        return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
      }
      if (pto::isPTOFloat8E5M2LikeType(type)) {
        return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
      }
      if (type.isF32()) {
        return emitc::OpaqueType::get(Ctx, "float");
      }
      if (type.isF16()) {
        return emitc::OpaqueType::get(Ctx, "half");
      }
      if (type.isBF16()) {
        return emitc::OpaqueType::get(Ctx, "bfloat16_t");
      }
      if (type.isF64()) {
        return emitc::OpaqueType::get(Ctx, "double");
      }
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](pto::HiF8Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "hifloat8_t");
    });
    addConversion([Ctx](Type type) -> std::optional<Type> {
      if (isF8E8M0ElemType(type)) {
        return emitc::OpaqueType::get(Ctx, "float8_e8m0_t");
      }
      return std::nullopt;
    });
    addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
    });
    addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
    });

    addConversion([Ctx](IntegerType type) -> Type {
      if (type.getWidth() == 1) {
        return type;
      }

      // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
      // explicitly signed/unsigned; treat signless as signed by default.
      const bool isUnsigned = type.isUnsignedInteger();
      switch (type.getWidth()) {
      case 8:
        return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
      case 16:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint16_t" : "int16_t");
      case 32:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint32_t" : "int32_t");
      case 64:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint64_t" : "int64_t");
      default:
        llvm::errs() << "[Debug] Unsupported IntegerType width: "
                     << type.getWidth() << "\n";
        return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
      }
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int64_t");
    });

    // vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == 1 && type.getNumElements() == 4 &&
          type.getElementType().isInteger(16)) {
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      }
      return Type{};
    });

    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](emitc::PointerType type) { return type; });

    // ---------------------------------------------------------
    // 2.5 PtrType 转换 (指针类型)
    // ---------------------------------------------------------
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType);
      if (!newElemType) {
        return std::nullopt;
      }

      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier =
          addrSpaceQualifier(getAddressSpaceOrGM(type.getMemorySpace()));

      return getEmitCPointerType(Ctx, qualifier, elemTypeStr);
    });

    addConversion([Ctx](pto::PipeType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "auto");
    });

    addConversion([Ctx](pto::EventIdArrayType type) -> Type {
      std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
      return emitc::OpaqueType::get(Ctx, tok);
    });

    // !pto.local_array<D1 x D2 x ... x T> -> !emitc.array<D1 x D2 x ... x T>.
    // Variables of this type render as `T a[D1][D2]...;` in the emitted C++.
    addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
      Type convertedElem = convertType(type.getElementType());
      if (!convertedElem) {
        return std::nullopt;
      }
      return emitc::ArrayType::get(type.getShape(), convertedElem);
    });

    // !pto.struct<...> -> !emitc.opaque<"PtoStruct_...">. The matching C++
    // `struct PtoStruct_... { ... };` definition is emitted at file scope by
    // the pass (see runOnOperation), keyed on the same content-derived name.
    // A struct is carried as a pointer to its storage. It cannot be carried by
    // value (emitc.member needs an lvalue, so every field write would land in a
    // copy), and it cannot be carried as an lvalue either: emitc.func rejects
    // an lvalue argument outright, and the C++ emitter refuses one on func.func
    // too, which would make a struct impossible to pass to a helper function.
    // A pointer is legal in a signature and still names the caller's storage.
    addConversion([Ctx](pto::StructType type) -> Type {
      return emitc::PointerType::get(
          emitc::OpaqueType::get(Ctx, getStructTypeName(type)));
    });

    addConversion([Ctx](pto::AsyncSessionType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
    });

    addConversion([Ctx](pto::AsyncEventType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
    });

    addConversion([Ctx](pto::PrefetchAsyncContextType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::PrefetchAsyncContext");
    });

    addConversion([Ctx](pto::TensorViewType type) -> Type {
      return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                              type.getShape());
    });

    addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
      return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                              type.getShape());
    });

    addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
      auto typeString = getEmitCTileTypeString(type);
      if (!typeString) {
        return std::nullopt;
      }
      return emitc::OpaqueType::get(Ctx, *typeString);
    });

    // ---------------------------------------------------------
    // 3. MemRef 转换 (Debug 重点)
    // ---------------------------------------------------------
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

      // A. 转换元素类型
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType); 
      if (!newElemType) {
        llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
        return std::nullopt;
      }
      
      // 获取元素类型的字符串
      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
         llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
         return std::nullopt;
      }

      // B. 处理 Memory Space
      std::string qualifier = "";
      Attribute memorySpace = type.getMemorySpace();
      
      if (!memorySpace) {
         qualifier = "__gm__";
      } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
         qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else {
         llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
         qualifier = "__gm__"; // Fallback
      }

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
      
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    // ---------------------------------------------------------
    // 4. Function & Materialization
    // ---------------------------------------------------------
    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      if (failed(convertTypes(type.getInputs(), inputs))) {
        return Type{};
      }
      SmallVector<Type> results;
      if (failed(convertTypes(type.getResults(), results))) {
        return Type{};
      }
      return FunctionType::get(type.getContext(), inputs, results);
    });

    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1) {
        return Value();
      }
      return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

static constexpr unsigned kPTOIndexBitWidth =
    64; // keep consistent with IndexType conversion

// Forward declarations (definitions below).
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a);
static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth);
static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth);
static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth);
static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth);
static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal);
static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value);
static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src);
static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr);
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth);
static bool needsA5NoSplitVectorGuard(Operation *op);

static FailureOr<std::string> getTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return std::string("TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT");
  case 3:
    return std::string("TileSplitAxis::TILE_UP_DOWN_ODD");
  case 4:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT_ODD");
  default:
    return failure();
  }
}

static FailureOr<std::string>
getTPipeDirectionToken(bool isL2G2L, int8_t dirMask, PTOArch targetArch) {
  if (dirMask == 1) {
    if (isL2G2L && targetArch == PTOArch::A5) {
      return std::string("Direction::DIR_C2V_GM");
    }
    return std::string("Direction::DIR_C2V");
  }
  if (dirMask == 2) {
    if (isL2G2L && targetArch == PTOArch::A5) {
      return std::string("Direction::DIR_V2C_GM");
    }
    return std::string("Direction::DIR_V2C");
  }
  if (dirMask == 3) {
    return std::string("Direction::DIR_BOTH");
  }
  return failure();
}

static std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok,
                                   int32_t slotSize, int32_t slotNum,
                                   int32_t localSlotNum, bool nosplit) {
  std::string token = "TPipe<" + std::to_string(flagBase) + ", " + dirTok.str() +
                      ", " + std::to_string(slotSize) + ", " +
                      std::to_string(slotNum);
  token += ", " + std::to_string(localSlotNum);
  token += nosplit ? ", true" : ", false";
  token += ">";
  return token;
}

static FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                        PTOArch targetArch) {
  if (auto initOp = dyn_cast<pto::InitializeL2G2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr()) {
      return failure();
    }
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/true, initOp.getDirMask(), targetArch);
    if (failed(dirTok)) {
      return failure();
    }
    int32_t localSlotNum =
        initOp.getLocalSlotNumAttr()
            ? static_cast<int32_t>(
                  getIntegerAttrSignedValue(initOp.getLocalSlotNumAttr()))
            : initOp.getSlotNum();
    return buildTPipeToken(
        static_cast<int32_t>(getIntegerAttrSignedValue(initOp.getFlagBaseAttr())),
        *dirTok, initOp.getSlotSize(), initOp.getSlotNum(), localSlotNum,
        initOp.getNosplitAttr() && initOp.getNosplitAttr().getValue());
  }

  if (auto initOp = dyn_cast<pto::InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr()) {
      return failure();
    }
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/false, initOp.getDirMask(), targetArch);
    if (failed(dirTok)) {
      return failure();
    }
    return buildTPipeToken(
        static_cast<int32_t>(getIntegerAttrSignedValue(initOp.getFlagBaseAttr())),
        *dirTok, initOp.getSlotSize(), initOp.getSlotNum(), 2,
        initOp.getNosplitAttr() && initOp.getNosplitAttr().getValue());
  }

  return failure();
}

static std::string buildFixpipeConfigAliasName(int32_t pipeId) {
  return "Pipe" + std::to_string(pipeId) + "FixpipeConfig";
}

static FailureOr<std::string> getFixpipeLayoutToken(FixpipeLayout layout) {
  switch (layout) {
  case FixpipeLayout::NZ2ND:
    return std::string("LayoutMode_t::NZ2ND");
  case FixpipeLayout::NZ2DN:
    return std::string("LayoutMode_t::NZ2DN");
  case FixpipeLayout::NZ2NZ:
    return std::string("LayoutMode_t::NZ2NZ");
  }
  return failure();
}

static FailureOr<std::string> getFixpipeQuantToken(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return std::string("QuantMode_t::NoQuant");
  case FixpipeQuant::F32F16:
    return std::string("QuantMode_t::F322F16");
  case FixpipeQuant::F32BF16:
    return std::string("QuantMode_t::F322BF16");
  case FixpipeQuant::REQ8Scalar:
    return std::string("QuantMode_t::REQ8");
  case FixpipeQuant::REQ8Vec:
    return std::string("QuantMode_t::VREQ8");
  case FixpipeQuant::DEQF16Scalar:
    return std::string("QuantMode_t::DEQF16");
  case FixpipeQuant::DEQF16Vec:
    return std::string("QuantMode_t::VDEQF16");
  case FixpipeQuant::QF322B8PreScalar:
    return std::string("QuantMode_t::QF322B8_PRE");
  case FixpipeQuant::QF322B8PreVec:
    return std::string("QuantMode_t::VQF322B8_PRE");
  case FixpipeQuant::QF322F16PreScalar:
    return std::string("QuantMode_t::QF322F16_PRE");
  case FixpipeQuant::QF322BF16PreScalar:
    return std::string("QuantMode_t::QF322BF16_PRE");
  case FixpipeQuant::QS322BF16PreScalar:
    return std::string("QuantMode_t::QS322BF16_PRE");
  case FixpipeQuant::QS322BF16PreVec:
    return std::string("QuantMode_t::VQS322BF16_PRE");
  case FixpipeQuant::QF322HIF8PreScalar:
    return std::string("QuantMode_t::QF322HIF8_PRE");
  case FixpipeQuant::QF322FP8PreScalar:
    return std::string("QuantMode_t::QF322FP8_PRE");
  }
  return failure();
}

static FailureOr<std::string> getFixpipeReluToken(FixpipeRelu relu) {
  switch (relu) {
  case FixpipeRelu::NoRelu:
    return std::string("ReluPreMode::NoRelu");
  case FixpipeRelu::NormalRelu:
    return std::string("ReluPreMode::NormalRelu");
  }
  return failure();
}

static FailureOr<std::string>
buildFixpipeConfigTypeToken(AccPushEpilogueAttr accPushEpilogue) {
  auto layoutTok = getFixpipeLayoutToken(accPushEpilogue.getLayout());
  auto quantTok = getFixpipeQuantToken(accPushEpilogue.getQuant());
  auto reluTok = getFixpipeReluToken(accPushEpilogue.getRelu());
  if (failed(layoutTok) || failed(quantTok) || failed(reluTok)) {
    return failure();
  }
  return "FixpipeParams<" + *layoutTok + ", " + *quantTok + ", " + *reluTok +
         ">";
}

static FailureOr<Operation *> findPeerFixpipeConsumerInit(Operation *producerInit) {
  auto ownerFuncAttr =
      producerInit->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveNameAttr =
      producerInit->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      producerInit->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerFuncAttr || !reserveNameAttr || !dirMaskAttr ||
      dirMaskAttr.getInt() != 1) {
    return failure();
  }

  auto peerFunc =
      lookupPeerFuncAcrossContainer(producerInit, ownerFuncAttr);
  if (!peerFunc) {
    return failure();
  }

  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerFunc.walk([&matchedInit, &matchedInitCount, ownerFuncAttr,
                 reserveNameAttr, dirMaskAttr, peerFunc](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate)) {
      return WalkResult::advance();
    }

    if (!getPipeInitAccPushEpilogue(candidate)) {
      return WalkResult::advance();
    }

    auto candidateOwnerFuncAttr =
        candidate->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
    auto candidateReserveNameAttr =
        candidate->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
    auto candidateDirMaskAttr =
        candidate->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
    if (!candidateOwnerFuncAttr || !candidateReserveNameAttr ||
        !candidateDirMaskAttr) {
      return WalkResult::advance();
    }

    if (candidateOwnerFuncAttr != ownerFuncAttr ||
        candidateReserveNameAttr != reserveNameAttr ||
        candidateDirMaskAttr.getInt() != dirMaskAttr.getInt()) {
      return WalkResult::advance();
    }

    auto candidateFunc = candidate->getParentOfType<func::FuncOp>();
    if (!candidateFunc || candidateFunc != peerFunc) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });
  if (matchedInitCount != 1 || !matchedInit) {
    return failure();
  }
  return matchedInit;
}

static FailureOr<TileBufType> resolveFixpipeConsumerTileType(Value pipeHandle) {
  Operation *producerInit = getPipeInitDef(pipeHandle);
  if (!producerInit) {
    return failure();
  }

  Type resolvedType;
  bool hasMismatch = false;

  auto collectFromFunc = [&resolvedType, &hasMismatch](
      func::FuncOp funcOp, llvm::function_ref<bool(pto::TPopOp)> matchesPop) {
    funcOp.walk([&resolvedType, &hasMismatch, &matchesPop](pto::TPopOp pop) {
      if (!matchesPop(pop)) {
        return WalkResult::advance();
      }
      if (!resolvedType) {
        resolvedType = pop.getTile().getType();
        return WalkResult::advance();
      }
      if (resolvedType != pop.getTile().getType()) {
        hasMismatch = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  };

  auto peerInitOr = findPeerFixpipeConsumerInit(producerInit);
  if (failed(peerInitOr)) {
    return failure();
  }

  Value peerPipe = (*peerInitOr)->getResult(0);
  collectFromFunc((*peerInitOr)->getParentOfType<func::FuncOp>(),
                  [&](pto::TPopOp pop) {
                    return peelUnrealized(pop.getPipeHandle()) == peerPipe;
                  });

  if (hasMismatch || !resolvedType) {
    return failure();
  }
  auto tileTy = dyn_cast<TileBufType>(resolvedType);
  if (!tileTy) {
    return failure();
  }
  return tileTy;
}

static LogicalResult rematerializeFixpipeQuantBindingsInBlock(
    Block &block, SmallVectorImpl<Operation *> &eraseList) {
  llvm::DenseMap<int32_t, SetQuantScalarOp> activeScalarById;
  llvm::DenseMap<int32_t, SetQuantVectorOp> activeVectorById;
  SmallVector<Operation *> originalOps;
  for (Operation &op : block) {
    originalOps.push_back(&op);
  }

  for (Operation *op : originalOps) {
    if (auto setQuantScalar = dyn_cast<SetQuantScalarOp>(op)) {
      activeScalarById[setQuantScalar.getId()] = setQuantScalar;
      eraseList.push_back(op);
    } else if (auto setQuantVector = dyn_cast<SetQuantVectorOp>(op)) {
      activeVectorById[setQuantVector.getId()] = setQuantVector;
      eraseList.push_back(op);
    } else if (auto tpush = dyn_cast<TPushOp>(op)) {
      auto accPushEpilogue =
          getPipeInitAccPushEpilogue(getPipeInitDef(tpush.getPipeHandle()));
      auto pipeId = getFrontendPipeIdFromHandle(tpush.getPipeHandle());
      if (accPushEpilogue && pipeId) {
        OpBuilder builder(tpush);
        if (isScalarFixpipeQuant(accPushEpilogue.getQuant())) {
          auto it = activeScalarById.find(*pipeId);
          if (it != activeScalarById.end()) {
            auto consumerTileTy =
                resolveFixpipeConsumerTileType(tpush.getPipeHandle());
            if (failed(consumerTileTy)) {
              tpush.emitOpError("failed to resolve peer consumer tile type "
                                "for fixpipe quant rematerialization");
              return failure();
            }
            Operation *cloned = builder.clone(*it->second.getOperation());
            cloned->setAttr(kEmitCScalarOutTypeAttrName,
                            builder.getStringAttr(getEmitCScalarTypeToken(
                                (*consumerTileTy).getElementType())));
          }
        } else if (isVectorFixpipeQuant(accPushEpilogue.getQuant())) {
          auto it = activeVectorById.find(*pipeId);
          if (it != activeVectorById.end()) {
            builder.clone(*it->second.getOperation());
          }
        }
      }
    }

    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        if (failed(rematerializeFixpipeQuantBindingsInBlock(nestedBlock,
                                                           eraseList))) {
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult rematerializeFixpipeQuantBindings(ModuleOp mop) {
  SmallVector<Operation *> eraseList;
  for (auto funcOp : mop.getOps<func::FuncOp>()) {
    for (Block &block : funcOp.getBlocks()) {
      if (failed(rematerializeFixpipeQuantBindingsInBlock(block, eraseList))) {
        return failure();
      }
    }
  }

  for (Operation *op : eraseList) {
    op->erase();
  }
  return success();
}

static LogicalResult insertFixpipeConfigAliases(ModuleOp mop) {
  for (auto funcOp : mop.getOps<func::FuncOp>()) {
    llvm::DenseSet<int32_t> seenIds;
    SmallVector<std::pair<int32_t, std::string>> aliases;
    bool aliasBuildFailed = false;
    funcOp.walk([&](TPushOp tpush) {
      auto accPushEpilogue = getPipeInitAccPushEpilogue(getPipeInitDef(tpush.getPipeHandle()));
      auto pipeId = getFrontendPipeIdFromHandle(tpush.getPipeHandle());
      if (!accPushEpilogue || !pipeId || !seenIds.insert(*pipeId).second) {
        return WalkResult::advance();
      }
      auto configTok = buildFixpipeConfigTypeToken(accPushEpilogue);
      if (failed(configTok)) {
        aliasBuildFailed = true;
        return WalkResult::interrupt();
      }
      aliases.emplace_back(*pipeId, *configTok);
      return WalkResult::advance();
    });
    if (aliasBuildFailed) {
      return failure();
    }

    if (aliases.empty()) {
      continue;
    }

    if (funcOp.empty()) {
      funcOp.emitError("cannot insert fixpipe config aliases into an external "
                       "function");
      return failure();
    }

    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.front());
    for (const auto &[pipeId, configTok] : aliases) {
      std::string line =
          "using " + buildFixpipeConfigAliasName(pipeId) + " = " + configTok + ";";
      builder.create<emitc::VerbatimOp>(
          funcOp.getLoc(), builder.getStringAttr(line));
    }
  }
  return success();
}

static FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle,
                                                     PTOArch targetArch) {
  pipeHandle = peelUnrealized(pipeHandle);
  Operation *def = pipeHandle.getDefiningOp();
  if (!def) {
    return failure();
  }
  return buildTPipeTokenFromInitOp(def, targetArch);
}

static bool isSetFFTsPointerLikeType(Type ty) {
  return isEmitCPointerLikeType(ty);
}

static bool tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

static Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as,
                                  StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as)) {
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
  return getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
}

static Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tile,
                                      pto::AddressSpace as,
                                      StringRef elemTok) {
  auto rawTy = getTileDataResultType(rewriter.getContext(), as, elemTok);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

static Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                         Location loc, Value addr,
                                         pto::AddressSpace as,
                                         StringRef elemTok) {
  auto *ctx = rewriter.getContext();
  std::string ptrTyStr =
      std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*";
  auto ptrTy = getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
  if (isSetFFTsPointerLikeType(addr.getType())) {
    if (addr.getType() == ptrTy) {
      return addr;
    }
    return rewriter.create<emitc::CastOp>(loc, ptrTy, addr).getResult();
  }
  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, ptrTyStr)});
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "reinterpret_cast",
                                   ArrayAttr{}, castTyAttr,
                                   ValueRange{addr})
      .getResult(0);
}

static bool isEmitCTileLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  if (!opaqueTy) {
    return false;
  }
  StringRef value = opaqueTy.getValue();
  return value.contains("Tile<") || value.contains("ConvTile<");
}

static FailureOr<Value> materializeCallOperandForEmitC(
    const TypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    Location loc, Type originalCalleeArgTy, Value tileLike,
    pto::AddressSpace addressSpace, StringRef elemTok) {
  Value extracted = materializeTileDataValue(
      rewriter, loc, tileLike, addressSpace, elemTok);
  if (!typeConverter) {
    return extracted;
  }
  Type targetTy = typeConverter->convertType(originalCalleeArgTy);
  if (!targetTy) {
    return failure();
  }
  const bool needsNoCast = extracted.getType() == targetTy;
  if (needsNoCast) {
    return extracted;
  }
  return rewriter.create<emitc::CastOp>(loc, targetTy, extracted).getResult();
}

static FailureOr<Value>
adaptCallOperandForEmitC(const TypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Type originalCalleeArgTy, Value originalOperand,
                         Value loweredOperand) {
  Type elemTy;
  std::optional<pto::AddressSpace> as;
  if (auto ptrTy = dyn_cast<pto::PtrType>(originalCalleeArgTy)) {
    elemTy = ptrTy.getElementType();
    as = getAddressSpaceOrGM(ptrTy.getMemorySpace());
  } else if (auto memrefTy = dyn_cast<MemRefType>(originalCalleeArgTy)) {
    elemTy = memrefTy.getElementType();
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(memrefTy.getMemorySpace())) {
      as = asAttr.getAddressSpace();
    } else {
      as = pto::AddressSpace::GM;
    }
  }

  if (elemTy && as) {
    std::string elemTokStorage = getEmitCScalarTypeToken(elemTy);
    StringRef elemTok(elemTokStorage);

    if (auto tileBufAddr = originalOperand.getDefiningOp<pto::TileBufAddrOp>()) {
      Value tileValue = loweredOperand;
      if (!isEmitCTileLikeType(tileValue.getType()) && tileBufAddr.getSrc()) {
        tileValue = tileBufAddr.getSrc();
      }
      if (isEmitCTileLikeType(tileValue.getType())) {
        return materializeCallOperandForEmitC(
            typeConverter, rewriter, loc, originalCalleeArgTy, tileValue, *as,
            elemTok);
      }
    }

    if (isEmitCTileLikeType(loweredOperand.getType())) {
      return materializeCallOperandForEmitC(
          typeConverter, rewriter, loc, originalCalleeArgTy, loweredOperand, *as,
          elemTok);
    }
  }

  return loweredOperand;
}

struct InterCoreSyncCallDesc {
  const char *callee = nullptr;
  ArrayAttr args;
  SmallVector<Value, 2> operands;
};

static Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty) {
    return eventId;
  }
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

static Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == 2) {
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  }
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

static Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventId, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventId})
      .getResult(0);
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCallImpl(
    ConversionPatternRewriter &rewriter, Value msgVal, PTOArch targetArch,
    pto::PipeAttr pipeAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  (void)targetArch;
  InterCoreSyncCallDesc desc;
  desc.callee = "__builtin_cce_ffts_cross_core_sync";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(msgVal);
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto indexTy = emitc::OpaqueType::get(rewriter.getContext(), "int64_t");
  Value eventVal =
      makeEmitCIntConstant(rewriter, loc, indexTy,
                           getIntegerAttrSignedValue(eventIdAttr));
  Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  Value msgVal = createFFTSMsg(rewriter, loc, eventIdVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
  desc.operands.push_back(castInterCoreEventIdToI32(rewriter, loc, eventIdVal));
  return desc;
}

static bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp, pto::SetCrossBlockOp,
            pto::WaitCrossBlockOp, pto::SetIntraBlockOp,
            pto::WaitIntraBlockOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

#include "PTOToEmitCArith.inc"
#include "PTOToEmitCMemref.inc"
#include "PTOToEmitCPointerAndMatmul.inc"
#include "PTOToEmitCSyncAndScalar.inc"
#include "PTOToEmitCAsyncAndComm.inc"
#include "PTOToEmitCDataObjects.inc"
#include "PTOToEmitCCompute.inc"
#include "PTOToEmitCPatternRegistration.inc"
#include "PTOToEmitCPass.inc"
