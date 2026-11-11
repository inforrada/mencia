import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import '../constants/app_constants.dart';

/// Normalization range applied to resized face tensors.
enum NormalizationRange {
  /// Scales pixel values from `[0, 255]` to `[0.0, 1.0]`.
  zeroToOne,

  /// Scales pixel values from `[0, 255]` to `[-1.0, 1.0]`.
  minusOneToOne,
}

/// Captures 3D head pose angles, relative frame position, bounding-box aspect
/// ratio, and high-resolution contour/landmark/classification cues from ML Kit
/// Face Detection.
class FacialPositionDescriptor {
  const FacialPositionDescriptor({
    this.headPitchX = 0.0,
    this.headYawY = 0.0,
    this.headRollZ = 0.0,
    this.normalizedCenterX = 0.0,
    this.normalizedCenterY = 0.0,
    this.boxAspectRatio = 0.85,
    this.smilingProbability,
    this.leftEyeOpenProbability,
    this.rightEyeOpenProbability,
    this.mouthCornerSlope = 0.0,
    this.mouthApertureRatio = 0.0,
    this.eyebrowArchRatio = 0.0,
    this.eyeAspectRatio = 0.0,
    this.lipContourAperture = 0.0,
    this.contourPointsCount = 0,
  });

  /// Head pitch angle in degrees (positive = looking up, negative = looking down).
  final double headPitchX;

  /// Head yaw angle in degrees (turn left / right).
  final double headYawY;

  /// Head roll angle in degrees (lateral tilt).
  final double headRollZ;

  /// Horizontal face center offset from frame center `[-0.5, 0.5]`.
  final double normalizedCenterX;

  /// Vertical face center offset from frame center `[-0.5, 0.5]`
  /// (positive = lower in frame, negative = higher in frame).
  final double normalizedCenterY;

  /// Ratio `boundingBox.width / boundingBox.height`.
  final double boxAspectRatio;

  /// ML Kit smiling probability `[0.0, 1.0]`, if available.
  final double? smilingProbability;

  /// ML Kit left eye open probability `[0.0, 1.0]`, if available.
  final double? leftEyeOpenProbability;

  /// ML Kit right eye open probability `[0.0, 1.0]`, if available.
  final double? rightEyeOpenProbability;

  /// Relative elevation of mouth corners vs lower lip center derived from landmarks/contours.
  final double mouthCornerSlope;

  /// Relative vertical mouth opening derived from nose/mouth landmarks.
  final double mouthApertureRatio;

  /// Relative eyebrow arch elevation above the eyes derived from dense contours.
  final double eyebrowArchRatio;

  /// Normalized eye vertical-to-horizontal openness ratio from dense eye contours.
  final double eyeAspectRatio;

  /// Normalized inner lip vertical separation from upper/lower lip contours.
  final double lipContourAperture;

  /// Total number of dense facial contour points detected on the face.
  final int contourPointsCount;
}

/// Represents a user-defined or training-validated mood sample with its
/// extracted facial feature vector used as a reference prototype.
class ValidatedMoodReference {
  const ValidatedMoodReference({
    required this.emotion,
    required this.featureVector,
    this.weight = 1.0,
  });

  final String emotion;
  final List<double> featureVector;

  /// Relative authority weight of this reference (e.g., 2.5 for explicitly
  /// configured Mood reference photos in Settings, 1.0 for training samples).
  final double weight;
}

/// Offline image preprocessing utilities for decoding, cropping with padding,
/// resizing, normalizing face regions, and computing position-aware &
/// reference-calibrated emotion probabilities.
class ImageUtils {
  ImageUtils._();

  /// Decodes raw compressed image bytes (JPEG/PNG) into an [img.Image] bitmap,
  /// baking EXIF orientation so front-camera photos are upright.
  static img.Image? decodeBytes(Uint8List bytes) {
    final img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) {
      return null;
    }
    return img.bakeOrientation(decoded);
  }

  /// Crops [sourceImage] to the specified face bounding box with
  /// [paddingRatio] (default 10%) clamped safely within image boundaries.
  static img.Image cropFaceWithPadding(
    img.Image sourceImage, {
    required double left,
    required double top,
    required double width,
    required double height,
    double paddingRatio = AppConstants.faceCropPaddingRatio,
  }) {
    final double padX = width * paddingRatio;
    final double padY = height * paddingRatio;

    final int cropX =
        math.max(0, (left - padX).floor()).clamp(0, sourceImage.width - 1);
    final int cropY =
        math.max(0, (top - padY).floor()).clamp(0, sourceImage.height - 1);

    final int maxRight =
        math.min(sourceImage.width, (left + width + padX).ceil());
    final int maxBottom =
        math.min(sourceImage.height, (top + height + padY).ceil());

    final int cropWidth = math.max(1, maxRight - cropX);
    final int cropHeight = math.max(1, maxBottom - cropY);

    return img.copyCrop(
      sourceImage,
      x: cropX,
      y: cropY,
      width: cropWidth,
      height: cropHeight,
    );
  }

  /// Generates a PNG byte buffer of the preprocessed (resized & grayscale/RGB)
  /// face image so the user can visually inspect the treated image in Training Mode.
  static Uint8List encodeProcessedPreviewPng(
    img.Image croppedFace, {
    int targetSize = AppConstants.grayscaleInputSize,
    bool grayscale = true,
  }) {
    final img.Image resized = img.copyResize(
      croppedFace,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.cubic,
    );
    final img.Image processed = grayscale ? img.grayscale(resized) : resized;
    return img.encodePng(processed);
  }

  /// Normalizes a single channel byte `[0..255]` into the target float range.
  static double normalizeChannel(
    num channelValue, {
    NormalizationRange range = NormalizationRange.zeroToOne,
  }) {
    final double unit = (channelValue.clamp(0, 255)) / 255.0;
    switch (range) {
      case NormalizationRange.zeroToOne:
        return unit;
      case NormalizationRange.minusOneToOne:
        return (unit * 2.0) - 1.0;
    }
  }

  /// Resizes [faceImage] to `[targetSize, targetSize]` (default 48x48) and
  /// converts pixel intensities into a normalized 4D grayscale tensor
  /// `[1, targetSize, targetSize, 1]`.
  static List<List<List<List<double>>>> preprocessToGrayscaleTensor(
    img.Image faceImage, {
    int targetSize = AppConstants.grayscaleInputSize,
    NormalizationRange range = NormalizationRange.zeroToOne,
  }) {
    final img.Image resized = img.copyResize(
      faceImage,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.cubic,
    );
    final img.Image grayscale = img.grayscale(resized);

    return List<List<List<List<double>>>>.generate(
      1,
      (_) => List<List<List<double>>>.generate(
        targetSize,
        (int y) => List<List<double>>.generate(
          targetSize,
          (int x) {
            final img.Pixel pixel = grayscale.getPixel(x, y);
            final double value = normalizeChannel(pixel.r, range: range);
            return <double>[value];
          },
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  /// Resizes [faceImage] to `[targetSize, targetSize]` (default 224x224) and
  /// converts pixels into a normalized 4D RGB tensor
  /// `[1, targetSize, targetSize, 3]`.
  static List<List<List<List<double>>>> preprocessToRgbTensor(
    img.Image faceImage, {
    int targetSize = AppConstants.rgbInputSize,
    NormalizationRange range = NormalizationRange.minusOneToOne,
  }) {
    final img.Image resized = img.copyResize(
      faceImage,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.cubic,
    );

    return List<List<List<List<double>>>>.generate(
      1,
      (_) => List<List<List<double>>>.generate(
        targetSize,
        (int y) => List<List<double>>.generate(
          targetSize,
          (int x) {
            final img.Pixel pixel = resized.getPixel(x, y);
            return <double>[
              normalizeChannel(pixel.r, range: range),
              normalizeChannel(pixel.g, range: range),
              normalizeChannel(pixel.b, range: range),
            ];
          },
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  /// Extracts a 14-dimensional feature vector by scanning the cropped face on a
  /// high-resolution [scanResolution]x[scanResolution] grid (default 224x224
  /// with cubic interpolation and 3x3 Sobel directional gradients) combined with
  /// 3D facial position, landmarks, and dense contours from [position].
  static List<double> extractFacialFeatureVector(
    img.Image croppedFace, {
    FacialPositionDescriptor position = const FacialPositionDescriptor(),
    int scanResolution = AppConstants.highResFacialScanSize,
  }) {
    final int s = math.max(48, scanResolution);
    final img.Image grid = img.grayscale(
      img.copyResize(
        croppedFace,
        width: s,
        height: s,
        interpolation: img.Interpolation.cubic,
      ),
    );

    double sum = 0.0;
    double sumSq = 0.0;
    final int n = s * s;
    final List<double> lum = List<double>.filled(n, 0.0);

    for (int y = 0; y < s; y++) {
      for (int x = 0; x < s; x++) {
        final double v = grid.getPixel(x, y).r / 255.0;
        lum[y * s + x] = v;
        sum += v;
        sumSq += v * v;
      }
    }

    final double mean = sum / n;
    final double variance = math.max(0.0, (sumSq / n) - (mean * mean));
    final double stdDev = math.max(0.04, math.sqrt(variance));

    int scaleCoord(int c48) => ((c48 * s) / 48.0).round().clamp(0, s);

    double regionMean48(int x0, int y0, int x1, int y1) {
      final int sx0 = scaleCoord(x0);
      final int sy0 = scaleCoord(y0);
      final int sx1 = scaleCoord(x1);
      final int sy1 = scaleCoord(y1);
      double acc = 0.0;
      int count = 0;
      for (int y = sy0; y < sy1; y++) {
        for (int x = sx0; x < sx1; x++) {
          acc += (lum[y * s + x] - mean) / stdDev;
          count++;
        }
      }
      return count > 0 ? acc / count : 0.0;
    }

    // High-resolution 3x3 Sobel edge + micro-gradient energy in normalized coordinates
    double regionSobelGradient48(int x0, int y0, int x1, int y1) {
      final int sx0 = math.max(1, scaleCoord(x0));
      final int sy0 = math.max(1, scaleCoord(y0));
      final int sx1 = math.min(s - 1, scaleCoord(x1));
      final int sy1 = math.min(s - 1, scaleCoord(y1));
      double acc = 0.0;
      int count = 0;
      for (int y = sy0; y < sy1; y++) {
        for (int x = sx0; x < sx1; x++) {
          final double tl = lum[(y - 1) * s + (x - 1)];
          final double tc = lum[(y - 1) * s + x];
          final double tr = lum[(y - 1) * s + (x + 1)];
          final double ml = lum[y * s + (x - 1)];
          final double mr = lum[y * s + (x + 1)];
          final double bl = lum[(y + 1) * s + (x - 1)];
          final double bc = lum[(y + 1) * s + x];
          final double br = lum[(y + 1) * s + (x + 1)];

          final double gx =
              ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * 0.25;
          final double gy =
              ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * 0.25;
          acc += (gx.abs() + gy.abs()) / stdDev;
          count++;
        }
      }
      return count > 0 ? acc / count : 0.0;
    }

    // 0: Upper brow elevation intensity + dense contour eyebrow arch
    final double upperBrow =
        regionMean48(10, 6, 38, 14) + (position.eyebrowArchRatio * 1.4);
    // 1: Inner brow (glabella) furrow high-res Sobel gradient
    final double innerBrowGrad = regionSobelGradient48(18, 9, 30, 18);
    // 2: Eye openness contrast (left + right eye high-res boxes + contour aspect)
    final double eyeContrast = 0.5 *
            (regionSobelGradient48(10, 14, 21, 23) +
                regionSobelGradient48(27, 14, 38, 23)) +
        (position.eyeAspectRatio * 0.8);
    // 3: Cheek elevation relative to lower jaw
    final double cheekMean =
        0.5 * (regionMean48(7, 23, 19, 32) + regionMean48(29, 23, 41, 32));
    final double jawMean =
        0.5 * (regionMean48(8, 36, 18, 44) + regionMean48(30, 36, 40, 44));
    final double cheekRaise = cheekMean - jawMean;
    // 4: Nose bridge & upper lip nasolabial tension
    final double noseBridgeGrad = regionSobelGradient48(19, 20, 29, 32);
    // 5: Central mouth aperture & inner lip contour opening
    final double mouthCenterGrad = regionSobelGradient48(16, 33, 32, 43);
    final double mouthCenterDarkness = -regionMean48(18, 34, 30, 41);
    final double mouthAperture = 0.55 * mouthCenterGrad +
        0.35 * mouthCenterDarkness +
        position.mouthApertureRatio * 1.4 +
        position.lipContourAperture * 1.6;
    // 6: Mouth corner curvature (upper corners vs lower corners + landmark/contour slope)
    final double upperCorners =
        0.5 * (regionMean48(11, 31, 17, 36) + regionMean48(31, 31, 37, 36));
    final double lowerCorners =
        0.5 * (regionMean48(11, 37, 17, 43) + regionMean48(31, 37, 37, 43));
    final double mouthCurvature =
        (lowerCorners - upperCorners) + (position.mouthCornerSlope * 2.2);
    // 7: Bilateral hemiface asymmetry across the high-res scan grid
    final int asymY0 = scaleCoord(8);
    final int asymY1 = scaleCoord(44);
    final int asymX0 = scaleCoord(6);
    final int asymX1 = scaleCoord(24);
    double asymAcc = 0.0;
    int asymCount = 0;
    for (int y = asymY0; y < asymY1; y++) {
      for (int x = asymX0; x < asymX1; x++) {
        asymAcc += (lum[y * s + x] - lum[y * s + (s - 1 - x)]) / stdDev;
        asymCount++;
      }
    }
    final double asymmetry =
        (asymCount > 0 ? asymAcc / asymCount : 0.0).clamp(-1.5, 1.5);

    // 8..13: Facial position & ML Kit pose/probability/contour features
    final double normPitchX = (position.headPitchX / 20.0).clamp(-1.5, 1.5);
    final double normYawY = (position.headYawY / 25.0).clamp(-1.5, 1.5);
    final double normRollZ = (position.headRollZ / 20.0).clamp(-1.5, 1.5);

    final double smileCue = position.smilingProbability != null
        ? (position.smilingProbability! - 0.35) * 2.2 +
            (position.mouthCornerSlope * 0.6)
        : mouthCurvature * 0.8 + cheekRaise * 0.5;

    final double? leftEye = position.leftEyeOpenProbability;
    final double? rightEye = position.rightEyeOpenProbability;
    final double eyeOpenCue = (leftEye != null && rightEye != null)
        ? (((leftEye + rightEye) * 0.5) - 0.60) * 2.4 +
            (position.eyeAspectRatio * 0.7)
        : (eyeContrast - 0.35) * 1.8;

    final double positionAndAspectCue =
        ((position.boxAspectRatio - 0.84) * 2.5) -
            (position.normalizedCenterY * 1.2) +
            (position.normalizedCenterX.abs() * 0.4);

    return <double>[
      upperBrow,
      innerBrowGrad,
      eyeContrast,
      cheekRaise,
      noseBridgeGrad,
      mouthAperture,
      mouthCurvature,
      asymmetry,
      normPitchX,
      normYawY,
      normRollZ,
      smileCue,
      eyeOpenCue,
      positionAndAspectCue,
    ];
  }

  /// Computes calibrated 7-class emotion probabilities (`[Angry, Disgust, Fear,
  /// Happy, Sad, Surprise, Neutral]`) combining:
  /// 1. Centered TFLite model output (`tfliteScores`),
  /// 2. 3D facial position, pose, and spatial region logits from `featureVector`,
  /// 3. Similarity-weighted reference prototypes from `validatedReferences`
  ///    (moods previously confirmed or corrected by the user).
  static List<double> computePositionAndReferenceProbabilities({
    required List<double> tfliteScores,
    required List<double> featureVector,
    List<ValidatedMoodReference> validatedReferences =
        const <ValidatedMoodReference>[],
  }) {
    // Center TFLite scores around their mean so uniform outputs (e.g. 1/7 = 0.143)
    // contribute 0 bias instead of flattening the distribution.
    final List<double> centeredTflite = List<double>.filled(7, 0.0);
    if (tfliteScores.length == 7) {
      final double meanScore =
          tfliteScores.fold(0.0, (double a, double b) => a + b) / 7.0;
      for (int i = 0; i < 7; i++) {
        centeredTflite[i] = (tfliteScores[i] - meanScore) * 4.5;
      }
    }

    final double upperBrow = featureVector.isNotEmpty ? featureVector[0] : 0.0;
    final double innerBrowGrad =
        featureVector.length > 1 ? featureVector[1] : 0.0;
    final double eyeContrast =
        featureVector.length > 2 ? featureVector[2] : 0.0;
    final double cheekRaise = featureVector.length > 3 ? featureVector[3] : 0.0;
    final double noseBridgeGrad =
        featureVector.length > 4 ? featureVector[4] : 0.0;
    final double mouthAperture =
        featureVector.length > 5 ? featureVector[5] : 0.0;
    final double mouthCurvature =
        featureVector.length > 6 ? featureVector[6] : 0.0;
    final double asymmetry = featureVector.length > 7 ? featureVector[7] : 0.0;
    final double normPitchX = featureVector.length > 8 ? featureVector[8] : 0.0;
    final double normYawY = featureVector.length > 9 ? featureVector[9] : 0.0;
    final double normRollZ =
        featureVector.length > 10 ? featureVector[10] : 0.0;
    final double smileCue = featureVector.length > 11 ? featureVector[11] : 0.0;
    final double eyeOpenCue =
        featureVector.length > 12 ? featureVector[12] : 0.0;
    final double posAspectCue =
        featureVector.length > 13 ? featureVector[13] : 0.0;

    final double poseDeviation =
        normPitchX.abs() + normYawY.abs() * 0.7 + normRollZ.abs() * 0.8;

    // Position + spatial geometry logits for each of the 7 canonical classes:
    // [0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral]
    final List<double> logits = <double>[
      // 0: Angry — brow furrow + downward forward tilt + tense/compressed mouth
      centeredTflite[0] +
          1.25 * (innerBrowGrad - 0.30) -
          0.95 * normPitchX +
          0.45 * normYawY.abs() -
          0.70 * smileCue -
          0.35 * upperBrow,
      // 1: Disgust — nose bridge tension + head roll/asymmetry + narrowed eyes
      centeredTflite[1] +
          1.20 * (noseBridgeGrad - 0.30) +
          0.95 * asymmetry.abs() +
          0.85 * normRollZ.abs() -
          0.45 * eyeOpenCue -
          0.40 * smileCue,
      // 2: Fear — wide eyes + raised brows + slight backward/upward pitch + tense mouth
      centeredTflite[2] +
          1.10 * eyeOpenCue +
          0.75 * (eyeContrast - 0.30) +
          0.65 * normPitchX +
          0.55 * normYawY.abs() +
          0.45 * mouthAperture -
          0.75 * smileCue,
      // 3: Happy — smile cue + upturned mouth corners + raised cheeks + wider aspect
      centeredTflite[3] +
          1.85 * smileCue +
          1.15 * mouthCurvature +
          0.90 * cheekRaise +
          0.55 * posAspectCue +
          0.25 * normRollZ.abs(),
      // 4: Sad — downward head pitch + drooped mouth corners + lower eye openness
      centeredTflite[4] -
          1.35 * normPitchX -
          1.05 * mouthCurvature -
          0.85 * eyeOpenCue -
          0.55 * posAspectCue +
          0.45 * normRollZ.abs() -
          0.90 * smileCue,
      // 5: Surprise — upward head pitch + wide open eyes + high brows + open mouth
      centeredTflite[5] +
          1.30 * normPitchX +
          1.25 * eyeOpenCue +
          1.15 * mouthAperture +
          0.75 * upperBrow -
          0.30 * innerBrowGrad,
      // 6: Neutral — balanced centered pose + moderate muscle activation
      centeredTflite[6] +
          0.95 * (0.65 - poseDeviation) -
          0.75 * smileCue.abs() -
          0.55 * mouthAperture.abs() -
          0.45 * mouthCurvature.abs(),
    ];

    // Convert base logits into sharpened softmax probabilities (temperature = 0.55)
    final List<double> baseProbs = _softmaxWithTemperature(
      logits,
      temperature: 0.55,
    );

    // If no validated reference moods exist yet, return baseProbs
    if (validatedReferences.isEmpty || featureVector.isEmpty) {
      return baseProbs;
    }

    // Compute similarity against user-validated reference moods
    // Higher weights on facial position/pose & key expression cues (indices 5..13)
    const List<double> featureWeights = <double>[
      0.8, // upperBrow
      1.0, // innerBrowGrad
      1.0, // eyeContrast
      1.1, // cheekRaise
      0.9, // noseBridgeGrad
      1.3, // mouthAperture
      1.5, // mouthCurvature
      1.0, // asymmetry
      1.8, // normPitchX (head pitch position)
      1.6, // normYawY (head yaw position)
      1.6, // normRollZ (head roll position)
      1.8, // smileCue
      1.5, // eyeOpenCue
      1.4, // posAspectCue
    ];

    final List<double> refClassScores = List<double>.filled(7, 0.0);
    double maxSimilarity = 0.0;
    double totalSimilarity = 0.0;

    for (final ValidatedMoodReference ref in validatedReferences) {
      final int classIdx = AppConstants.emotionLabels.indexOf(ref.emotion);
      if (classIdx < 0 || ref.featureVector.isEmpty) {
        continue;
      }
      final int dim = math.min(featureVector.length, ref.featureVector.length);
      double weightedSqDist = 0.0;
      double weightSum = 0.0;
      for (int k = 0; k < dim; k++) {
        final double w = k < featureWeights.length ? featureWeights[k] : 1.0;
        final double diff = featureVector[k] - ref.featureVector[k];
        weightedSqDist += w * diff * diff;
        weightSum += w;
      }
      final double meanSqDist =
          weightSum > 0.0 ? weightedSqDist / weightSum : 1.0;
      final double similarity = math.exp(-3.5 * meanSqDist);
      final double effectiveWeight = ref.weight.clamp(0.1, 5.0);
      final double weightedSimilarity = similarity * effectiveWeight;
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
      }
      refClassScores[classIdx] += weightedSimilarity;
      totalSimilarity += weightedSimilarity;
    }

    if (totalSimilarity <= 1e-6) {
      return baseProbs;
    }

    final List<double> refProbs = refClassScores
        .map((double s) => s / totalSimilarity)
        .toList(growable: false);

    // Blend base probabilities with validated reference distribution.
    // The closer the current facial position/expression is to a configured or
    // validated reference, the stronger the reference pull (up to 85%).
    final double alpha = (0.25 + 0.60 * maxSimilarity).clamp(0.25, 0.85);
    final List<double> blended = List<double>.generate(
      7,
      (int i) => (1.0 - alpha) * baseProbs[i] + alpha * refProbs[i],
      growable: false,
    );
    final double blendedSum =
        blended.fold(0.0, (double prev, double elem) => prev + elem);
    return blended.map((double p) => p / blendedSum).toList(growable: false);
  }

  static List<double> _softmaxWithTemperature(
    List<double> logits, {
    double temperature = 1.0,
  }) {
    final double invTemp = 1.0 / math.max(0.05, temperature);
    final List<double> scaled =
        logits.map((double x) => x * invTemp).toList(growable: false);
    final double maxVal = scaled.reduce(math.max);
    final List<double> exps =
        scaled.map((double x) => math.exp(x - maxVal)).toList(growable: false);
    final double sum = exps.fold(0.0, (double a, double b) => a + b);
    return exps.map((double e) => e / sum).toList(growable: false);
  }

  /// Ensures a 7-class output vector represents valid softmax probabilities
  /// summing to `1.0`. If raw logits are passed, applies numerically stable softmax.
  static List<double> ensureSoftmaxProbabilities(List<double> rawScores) {
    if (rawScores.isEmpty) {
      return const <double>[];
    }
    double sum = 0.0;
    bool allInZeroOne = true;
    for (final double s in rawScores) {
      if (s < 0.0 || s > 1.0) {
        allInZeroOne = false;
      }
      sum += s;
    }
    if (allInZeroOne && (sum - 1.0).abs() <= 1e-2) {
      return List<double>.from(rawScores, growable: false);
    }

    return _softmaxWithTemperature(rawScores, temperature: 1.0);
  }

  /// Returns mood probability entries from [probabilities] whose score is at
  /// least [minProbability] (default 5%), sorted in descending order.
  /// If [excludeEmotion] is provided, that primary emotion is excluded so only
  /// secondary candidate moods are returned.
  static List<MapEntry<String, double>> filterSignificantMoods(
    Map<String, double> probabilities, {
    double minProbability = AppConstants.minSecondaryMoodProbability,
    String? excludeEmotion,
  }) {
    final List<MapEntry<String, double>> filtered = probabilities.entries
        .where(
          (MapEntry<String, double> entry) =>
              entry.value >= minProbability && entry.key != excludeEmotion,
        )
        .toList();
    filtered.sort(
      (MapEntry<String, double> a, MapEntry<String, double> b) =>
          b.value.compareTo(a.value),
    );
    return filtered;
  }

  /// Formats an image processing duration in milliseconds into a human-readable
  /// string showing seconds and milliseconds (e.g., `1.24 s (1240 ms)`).
  static String formatProcessingDuration(int durationMs) {
    final int safeMs = math.max(0, durationMs);
    final double seconds = safeMs / 1000.0;
    return '${seconds.toStringAsFixed(2)} s ($safeMs ms)';
  }
}
