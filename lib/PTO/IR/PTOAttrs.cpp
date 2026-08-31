// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"          // parseAttribute
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kI32BitWidth = 32;
constexpr int32_t kFractalSize512 = 512;
constexpr int32_t kBLayoutRowMajor = static_cast<int32_t>(BLayout::RowMajor);
constexpr int32_t kBLayoutColMajor = static_cast<int32_t>(BLayout::ColMajor);
constexpr int32_t kSLayoutNoneBox = static_cast<int32_t>(SLayout::NoneBox);
constexpr int32_t kSLayoutColMajor = static_cast<int32_t>(SLayout::ColMajor);
constexpr int32_t kPadValueNull = static_cast<int32_t>(PadValue::Null);
constexpr int32_t kPadValueMin = static_cast<int32_t>(PadValue::Min);
constexpr int32_t kCompactModeNull = static_cast<int32_t>(CompactMode::Null);
constexpr int32_t kCompactModeRowPlusOne =
    static_cast<int32_t>(CompactMode::RowPlusOne);

static LogicalResult parseTileBufKeyEq(AsmParser &parser,
                                       StringRef expectedKey) {
  if (failed(parser.parseKeyword(expectedKey))) {
    return failure();
  }
  return parser.parseEqual();
}

} // namespace

TileBufConfigAttr TileBufConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  BLayoutAttr bl = BLayoutAttr::get(ctx, BLayout::RowMajor);
  SLayoutAttr sl = SLayoutAttr::get(ctx, SLayout::NoneBox);
  PadValueAttr pv = PadValueAttr::get(ctx, PadValue::Null);
  CompactModeAttr compact = CompactModeAttr::get(ctx, CompactMode::Null);
  IntegerAttr sz = b.getI32IntegerAttr(kFractalSize512);
  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

bool TileBufConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getBLayout() == d.getBLayout() &&
         getSLayout() == d.getSLayout() &&
         getSFractalSize() == d.getSFractalSize() &&
         getPad() == d.getPad() &&
         getCompactMode() == d.getCompactMode();
}

static int32_t getLayoutInt(Attribute a, int32_t def) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) {
    return static_cast<int32_t>(bl.getValue());
  }
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) {
    return static_cast<int32_t>(sl.getValue());
  }
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) {
    return static_cast<int32_t>(pv.getValue());
  }
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) {
    return static_cast<int32_t>(cm.getValue());
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return static_cast<int32_t>(ia.getInt());
  }
  return def;
}

LogicalResult TileBufConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       Attribute bLayout,
                                       Attribute sLayout,
                                       IntegerAttr sFractalSize,
                                       Attribute pad,
                                       Attribute compactMode) {
  if (!bLayout || (!mlir::isa<BLayoutAttr>(bLayout) && !mlir::isa<IntegerAttr>(bLayout))) {
    return emitError() << "blayout must be BLayoutAttr or i32 integer attr", failure();
  }
  if (!sLayout || (!mlir::isa<SLayoutAttr>(sLayout) && !mlir::isa<IntegerAttr>(sLayout))) {
    return emitError() << "slayout must be SLayoutAttr or i32 integer attr", failure();
  }
  if (!pad || (!mlir::isa<PadValueAttr>(pad) && !mlir::isa<IntegerAttr>(pad))) {
    return emitError() << "pad must be PadValueAttr or i32 integer attr", failure();
  }
  if (!compactMode ||
      (!mlir::isa<CompactModeAttr>(compactMode) &&
       !mlir::isa<IntegerAttr>(compactMode))) {
    return emitError() << "compact_mode must be CompactModeAttr or i32 integer attr", failure();
  }

  if (!sFractalSize || !sFractalSize.getType().isInteger(kI32BitWidth)) {
    return emitError() << "s_fractal_size must be i32", failure();
  }

  int32_t s = static_cast<int32_t>(sFractalSize.getInt());
  if (s != kFractalMxSize && s != kFractalABSize && s != kFractalCSize) {
    return emitError() << "unsupported s_fractal_size: " << s
                       << ", must be one of {"
                       << kFractalMxSize << ", "
                       << kFractalABSize << ", "
                       << kFractalCSize << "}",
           failure();
  }

  int32_t blv = getLayoutInt(bLayout, -1);
  if (blv != kBLayoutRowMajor && blv != kBLayoutColMajor) {
    return emitError() << "unsupported blayout value: " << blv, failure();
  }

  int32_t slv = getLayoutInt(sLayout, -1);
  if (slv < kSLayoutNoneBox || slv > kSLayoutColMajor) {
    return emitError() << "unsupported slayout value: " << slv, failure();
  }

  int32_t pvv = getLayoutInt(pad, -1);
  if (pvv < kPadValueNull || pvv > kPadValueMin) {
    return emitError() << "unsupported pad value: " << pvv, failure();
  }

  int32_t cmv = getLayoutInt(compactMode, -1);
  if (cmv < kCompactModeNull || cmv > kCompactModeRowPlusOne) {
    return emitError() << "unsupported compact_mode value: " << cmv, failure();
  }

  return success();
}

// Helper: parse Attribute and convert to BLayoutAttr/SLayoutAttr/PadValueAttr
static BLayoutAttr toBLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) {
    return bl;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return BLayoutAttr::get(ctx, static_cast<BLayout>(ia.getInt()));
  }
  return {};
}
static SLayoutAttr toSLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) {
    return sl;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return SLayoutAttr::get(ctx, static_cast<SLayout>(ia.getInt()));
  }
  return {};
}
static PadValueAttr toPadValueAttr(MLIRContext *ctx, Attribute a) {
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) {
    return pv;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return PadValueAttr::get(ctx, static_cast<PadValue>(ia.getInt()));
  }
  return {};
}
static CompactModeAttr toCompactModeAttr(MLIRContext *ctx, Attribute a) {
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) {
    return cm;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return CompactModeAttr::get(ctx, static_cast<CompactMode>(ia.getInt()));
  }
  return {};
}

Attribute TileBufConfigAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();
  CompactModeAttr compact = def.getCompactMode();

  if (p.parseLess()) {
    return {};
  }

  if (succeeded(p.parseOptionalGreater())) {
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
  }

  bool parsedGreater = false;
  while (!parsedGreater) {
    StringRef key;
    if (p.parseKeyword(&key)) {
      return {};
    }
    if (p.parseEqual()) {
      return {};
    }

    if (key == "blayout") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      bl = toBLayoutAttr(ctx, a);
      if (!bl) {
        return {};
      }
    } else if (key == "slayout") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      sl = toSLayoutAttr(ctx, a);
      if (!sl) {
        return {};
      }
    } else if (key == "s_fractal_size") {
      int32_t v = 0;
      if (p.parseInteger(v)) {
        return {};
      }
      sz = IntegerAttr::get(IntegerType::get(ctx, kI32BitWidth), v);
    } else if (key == "pad") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      pv = toPadValueAttr(ctx, a);
      if (!pv) {
        return {};
      }
    } else if (key == "compact") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      compact = toCompactModeAttr(ctx, a);
      if (!compact) {
        return {};
      }
    } else {
      p.emitError(p.getCurrentLocation(), "unknown key in tile_buf_config: ") << key;
      return {};
    }

    parsedGreater = succeeded(p.parseOptionalGreater());
    if (parsedGreater) {
      break;
    }
    if (p.parseComma()) {
      return {};
    }
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

void TileBufConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "blayout=" << getBLayout();
  p << ", slayout=" << getSLayout();
  p << ", s_fractal_size=" << static_cast<int32_t>(getSFractalSize().getInt());
  p << ", pad=" << getPad();
  p << ", compact=" << getCompactMode();
  p << ">";
}

namespace {

constexpr unsigned kConvTilePadListSize = 4;
constexpr int32_t kConvTileDefaultZero = 0;
constexpr int32_t kConvTileDefaultOne = 1;

static LogicalResult parseTileBufKeyEq(AsmParser &parser, StringRef expectedKey);

static LogicalResult parseConvTileIntField(AsmParser &parser, StringRef key,
                                           IntegerAttr &value) {
  if (failed(parseTileBufKeyEq(parser, key))) {
    return failure();
  }
  int64_t parsed = 0;
  if (failed(parser.parseInteger(parsed))) {
    return failure();
  }
  value = IntegerAttr::get(IntegerType::get(parser.getContext(), kI32BitWidth),
                           parsed);
  return success();
}

static LogicalResult parseConvTileBoolField(AsmParser &parser, StringRef key,
                                            BoolAttr &value) {
  if (failed(parseTileBufKeyEq(parser, key))) {
    return failure();
  }
  bool parsed = false;
  if (succeeded(parser.parseOptionalKeyword("true"))) {
    parsed = true;
  } else if (succeeded(parser.parseOptionalKeyword("false"))) {
    parsed = false;
  } else {
    parser.emitError(parser.getCurrentLocation())
        << key << " must be true or false";
    return failure();
  }
  value = BoolAttr::get(parser.getContext(), parsed);
  return success();
}

static LogicalResult parseConvTileAttrField(AsmParser &parser, StringRef key,
                                            Attribute &value) {
  if (failed(parseTileBufKeyEq(parser, key))) {
    return failure();
  }
  if (failed(parser.parseAttribute(value))) {
    return failure();
  }
  return success();
}

static LogicalResult parseConvTilePadListField(AsmParser &parser,
                                               SmallVectorImpl<int64_t> &padList) {
  if (failed(parseTileBufKeyEq(parser, "pad_list"))) {
    return failure();
  }
  SmallVector<int64_t, kConvTilePadListSize> parsed;
  if (failed(parser.parseDimensionList(parsed, /*allowDynamic=*/false,
                                       /*withTrailingX=*/false))) {
    return failure();
  }
  if (parsed.size() != kConvTilePadListSize) {
    parser.emitError(parser.getCurrentLocation())
        << "pad_list must have exactly " << kConvTilePadListSize
        << " dimensions";
    return failure();
  }
  padList.assign(parsed.begin(), parsed.end());
  return success();
}

static void printConvTileIntField(AsmPrinter &p, StringRef key, IntegerAttr value) {
  p << key << "=" << value.getInt();
}

static void printConvTilePadList(AsmPrinter &p, ArrayRef<int64_t> padList) {
  for (auto it = padList.begin(); it != padList.end(); ++it) {
    if (it != padList.begin()) {
      p << "x";
    }
    p << *it;
  }
}

static StringRef printBoolAttr(BoolAttr value) {
  return value.getValue() ? "true" : "false";
}

} // namespace

ConvTileConfigAttr ConvTileConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  auto zero = b.getI32IntegerAttr(kConvTileDefaultZero);
  auto one = b.getI32IntegerAttr(kConvTileDefaultOne);
  SmallVector<int64_t, kConvTilePadListSize> padList(kConvTilePadListSize,
                                                     kConvTileDefaultZero);
  auto padValue = b.getI32IntegerAttr(kConvTileDefaultZero);
  auto transpose = BoolAttr::get(ctx, false);
  return ConvTileConfigAttr::get(ctx, zero, zero, padList, one, one, one, one,
                                 one, one, padValue, zero, zero, one, zero,
                                 zero, zero, transpose);
}

bool ConvTileConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getFmapH() == d.getFmapH() && getFmapW() == d.getFmapW() &&
         getPadList() == d.getPadList() && getFilterH() == d.getFilterH() &&
         getFilterW() == d.getFilterW() &&
         getDilationH() == d.getDilationH() &&
         getDilationW() == d.getDilationW() &&
         getStrideH() == d.getStrideH() && getStrideW() == d.getStrideW() &&
         getPadValue() == d.getPadValue() &&
         getChannelSize() == d.getChannelSize() &&
         getRepeatStride() == d.getRepeatStride() &&
         getRepeatTime() == d.getRepeatTime() &&
         getRepeatMode() == d.getRepeatMode() &&
         getDstStride() == d.getDstStride() &&
         getDstMposition() == d.getDstMposition() &&
         getTranspose() == d.getTranspose();
}

LogicalResult ConvTileConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                         IntegerAttr fmapH,
                                         IntegerAttr fmapW,
                                         ArrayRef<int64_t> padList,
                                         IntegerAttr filterH,
                                         IntegerAttr filterW,
                                         IntegerAttr dilationH,
                                         IntegerAttr dilationW,
                                         IntegerAttr strideH,
                                         IntegerAttr strideW,
                                         Attribute padValueAttr,
                                         IntegerAttr channelSize,
                                         IntegerAttr repeatStride,
                                         IntegerAttr repeatTime,
                                         IntegerAttr repeatMode,
                                         IntegerAttr dstStride,
                                         IntegerAttr dstMposition,
                                         BoolAttr transpose) {
  auto checkI32 = [&](IntegerAttr attr, StringRef name) -> LogicalResult {
    if (!attr || !attr.getType().isSignlessInteger(kI32BitWidth)) {
      return emitError() << name << " must be an i32 integer attr", failure();
    }
    return success();
  };

  if (failed(checkI32(fmapH, "fmap_h")) || failed(checkI32(fmapW, "fmap_w")) ||
      failed(checkI32(filterH, "filter_h")) ||
      failed(checkI32(filterW, "filter_w")) ||
      failed(checkI32(dilationH, "dilation_h")) ||
      failed(checkI32(dilationW, "dilation_w")) ||
      failed(checkI32(strideH, "stride_h")) ||
      failed(checkI32(strideW, "stride_w")) ||
      failed(checkI32(channelSize, "channel_size")) ||
      failed(checkI32(repeatStride, "repeat_stride")) ||
      failed(checkI32(repeatTime, "repeat_time")) ||
      failed(checkI32(repeatMode, "repeat_mode")) ||
      failed(checkI32(dstStride, "dst_stride")) ||
      failed(checkI32(dstMposition, "dst_mposition"))) {
    return failure();
  }

  if (!transpose) {
    return emitError() << "transpose must be a bool attr", failure();
  }

  if (padList.size() != kConvTilePadListSize) {
    return emitError() << "pad_list must have exactly 4 elements", failure();
  }
  for (int64_t dim : padList) {
    if (dim < 0 || dim > 255) {
      return emitError() << "pad_list entries must be in [0, 255]", failure();
    }
  }

  auto checkNonNegative = [&](IntegerAttr attr, StringRef name) -> LogicalResult {
    if (attr.getInt() < 0) {
      return emitError() << name << " must be non-negative", failure();
    }
    return success();
  };

  if (failed(checkNonNegative(fmapH, "fmap_h")) ||
      failed(checkNonNegative(fmapW, "fmap_w")) ||
      failed(checkNonNegative(filterH, "filter_h")) ||
      failed(checkNonNegative(filterW, "filter_w")) ||
      failed(checkNonNegative(dilationH, "dilation_h")) ||
      failed(checkNonNegative(dilationW, "dilation_w")) ||
      failed(checkNonNegative(strideH, "stride_h")) ||
      failed(checkNonNegative(strideW, "stride_w")) ||
      failed(checkNonNegative(channelSize, "channel_size")) ||
      failed(checkNonNegative(repeatStride, "repeat_stride")) ||
      failed(checkNonNegative(repeatTime, "repeat_time")) ||
      failed(checkNonNegative(repeatMode, "repeat_mode")) ||
      failed(checkNonNegative(dstStride, "dst_stride")) ||
      failed(checkNonNegative(dstMposition, "dst_mposition"))) {
    return failure();
  }

  if (!padValueAttr ||
      (!llvm::isa<IntegerAttr>(padValueAttr) &&
       !llvm::isa<FloatAttr>(padValueAttr))) {
    return emitError() << "pad_value must be an integer or float attr", failure();
  }

  return success();
}

Attribute ConvTileConfigAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto def = ConvTileConfigAttr::getDefault(ctx);
  IntegerAttr fmapH = def.getFmapH();
  IntegerAttr fmapW = def.getFmapW();
  SmallVector<int64_t, kConvTilePadListSize> padList(def.getPadList().begin(),
                                                     def.getPadList().end());
  IntegerAttr filterH = def.getFilterH();
  IntegerAttr filterW = def.getFilterW();
  IntegerAttr dilationH = def.getDilationH();
  IntegerAttr dilationW = def.getDilationW();
  IntegerAttr strideH = def.getStrideH();
  IntegerAttr strideW = def.getStrideW();
  Attribute padValue = def.getPadValue();
  IntegerAttr channelSize = def.getChannelSize();
  IntegerAttr repeatStride = def.getRepeatStride();
  IntegerAttr repeatTime = def.getRepeatTime();
  IntegerAttr repeatMode = def.getRepeatMode();
  IntegerAttr dstStride = def.getDstStride();
  IntegerAttr dstMposition = def.getDstMposition();
  BoolAttr transpose = def.getTranspose();
  bool parsedGreater = false;

  auto consumeFieldTerminator = [&]() -> LogicalResult {
    if (succeeded(p.parseOptionalGreater())) {
      parsedGreater = true;
      return success();
    }
    return p.parseComma();
  };

  if (p.parseLess()) {
    return {};
  }

  if (succeeded(p.parseOptionalGreater())) {
    return ConvTileConfigAttr::get(ctx, fmapH, fmapW, padList, filterH, filterW,
                                   dilationH, dilationW, strideH, strideW,
                                   padValue, channelSize, repeatStride,
                                   repeatTime, repeatMode, dstStride,
                                   dstMposition, transpose);
  }

  while (!parsedGreater) {
    StringRef key;
    if (p.parseKeyword(&key)) {
      return {};
    }
    if (p.parseEqual()) {
      return {};
    }

    if (key == "fmap_h") {
      if (failed(parseConvTileIntField(p, key, fmapH))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "fmap_w") {
      if (failed(parseConvTileIntField(p, key, fmapW))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "pad_list") {
      if (failed(parseConvTilePadListField(p, padList))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "filter_h") {
      if (failed(parseConvTileIntField(p, key, filterH))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "filter_w") {
      if (failed(parseConvTileIntField(p, key, filterW))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "dilation_h") {
      if (failed(parseConvTileIntField(p, key, dilationH))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "dilation_w") {
      if (failed(parseConvTileIntField(p, key, dilationW))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "stride_h") {
      if (failed(parseConvTileIntField(p, key, strideH))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "stride_w") {
      if (failed(parseConvTileIntField(p, key, strideW))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "pad_value") {
      if (failed(parseConvTileAttrField(p, key, padValue))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "channel_size") {
      if (failed(parseConvTileIntField(p, key, channelSize))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "repeat_stride") {
      if (failed(parseConvTileIntField(p, key, repeatStride))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "repeat_time") {
      if (failed(parseConvTileIntField(p, key, repeatTime))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "repeat_mode") {
      if (failed(parseConvTileIntField(p, key, repeatMode))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "dst_stride") {
      if (failed(parseConvTileIntField(p, key, dstStride))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "dst_mposition") {
      if (failed(parseConvTileIntField(p, key, dstMposition))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }
    if (key == "transpose") {
      if (failed(parseConvTileBoolField(p, key, transpose))) {
        return {};
      }
      if (failed(consumeFieldTerminator())) {
        return {};
      }
      continue;
    }

    p.emitError(p.getCurrentLocation(), "unknown key in conv_tile_config: ")
        << key;
    return {};
  }

  return ConvTileConfigAttr::get(ctx, fmapH, fmapW, padList, filterH, filterW,
                                 dilationH, dilationW, strideH, strideW,
                                 padValue, channelSize, repeatStride,
                                 repeatTime, repeatMode, dstStride, dstMposition,
                                 transpose);
}

void ConvTileConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  printConvTileIntField(p, "fmap_h", getFmapH());
  p << ", ";
  printConvTileIntField(p, "fmap_w", getFmapW());
  p << ", pad_list=";
  printConvTilePadList(p, getPadList());
  p << ", ";
  printConvTileIntField(p, "filter_h", getFilterH());
  p << ", ";
  printConvTileIntField(p, "filter_w", getFilterW());
  p << ", ";
  printConvTileIntField(p, "dilation_h", getDilationH());
  p << ", ";
  printConvTileIntField(p, "dilation_w", getDilationW());
  p << ", ";
  printConvTileIntField(p, "stride_h", getStrideH());
  p << ", ";
  printConvTileIntField(p, "stride_w", getStrideW());
  p << ", pad_value=" << getPadValue();
  p << ", ";
  printConvTileIntField(p, "channel_size", getChannelSize());
  p << ", ";
  printConvTileIntField(p, "repeat_stride", getRepeatStride());
  p << ", ";
  printConvTileIntField(p, "repeat_time", getRepeatTime());
  p << ", ";
  printConvTileIntField(p, "repeat_mode", getRepeatMode());
  p << ", ";
  printConvTileIntField(p, "dst_stride", getDstStride());
  p << ", ";
  printConvTileIntField(p, "dst_mposition", getDstMposition());
  p << ", transpose=" << printBoolAttr(getTranspose());
  p << ">";
}
