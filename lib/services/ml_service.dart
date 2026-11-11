import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../core/constants/app_constants.dart';
import '../core/utils/image_utils.dart';
import 'storage_service.dart';

/// Result of an offline facial emotion inference pass, including the
/// in-memory PNG preview of the preprocessed face (`processedImageBytes`),
/// the extracted 14-D facial position & appearance `featureVector`,
/// raw TFLite scores for reference re-evaluation, high-res scan size, and
/// total processing time (`processingDurationMs`).
class EmotionInferenceResult {
  const EmotionInferenceResult({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.timestamp,
    this.processedImageBytes,
    this.processedImageSize = AppConstants.defaultProcessedImageSize,
    this.modelInputSize = AppConstants.defaultInputSize,
    this.scanResolution = AppConstants.highResFacialScanSize,
    this.processingDurationMs = 0,
    this.featureVector = const <double>[],
    this.rawTfliteScores = const <double>[],
  });

  final String label;
  final double confidence;
  final Map<String, double> probabilities;
  final DateTime timestamp;

  /// Preprocessed (cropped, resized, grayscale/RGB) face image encoded as PNG
  /// in memory so the raw temporary camera file on disk can be deleted immediately.
  final Uint8List? processedImageBytes;

  /// Side length in pixels of [processedImageBytes] (user-configurable).
  final int processedImageSize;

  /// Side length in pixels of the tensor actually fed to the TFLite model,
  /// read from the model's input shape.
  final int modelInputSize;

  /// Side length in pixels of the high-resolution grid used to scan facial
  /// positions, contours, and Sobel gradients.
  final int scanResolution;

  /// Total time in milliseconds spent capturing and processing the image.
  final int processingDurationMs;

  /// Extracted facial position & spatial appearance feature vector.
  final List<double> featureVector;

  /// Raw 7-class scores produced by the TFLite interpreter.
  final List<double> rawTfliteScores;
}

/// Offline Machine Learning service coordinating Google ML Kit Face Detection
/// and a 2-thread CPU TensorFlow Lite emotion classification interpreter.
class MlService {
  FaceDetector? _faceDetector;
  Interpreter? _interpreter;
  String? _lastError;

  String? get lastError => _lastError;
  bool get isReady => _faceDetector != null && _interpreter != null;

  /// Initializes the ML Kit [FaceDetector] (with dense contours, landmarks,
  /// and classification enabled for high-precision facial position/pose and
  /// expression scanning) and loads the TFLite model using strictly 2 CPU
  /// worker threads.
  Future<bool> initialize() async {
    try {
      _lastError = null;
      _faceDetector ??= FaceDetector(
        options: FaceDetectorOptions(
          performanceMode: FaceDetectorMode.fast,
          enableContours: true,
          enableLandmarks: true,
          enableClassification: true,
        ),
      );

      final InterpreterOptions options = InterpreterOptions()
        ..threads = AppConstants.tfliteCpuThreads;

      _interpreter = await Interpreter.fromAsset(
        AppConstants.emotionModelAssetPath,
        options: options,
      );
      return true;
    } catch (e) {
      _lastError = 'Error al inicializar el motor de IA: $e';
      return false;
    }
  }

  /// Runs the offline face detection and emotion classification pipeline on
  /// [imagePath], combining TFLite spatial output, high-resolution 3D facial
  /// position/contours/landmarks, and similarity to [validatedReferences],
  /// while measuring processing time and guaranteeing deletion of the
  /// temporary photo file.
  Future<EmotionInferenceResult?> processImageFile(
    String imagePath, {
    NormalizationRange normalization = NormalizationRange.zeroToOne,
    int processedImageSize = AppConstants.defaultProcessedImageSize,
    int scanResolution = AppConstants.highResFacialScanSize,
    int captureDurationMs = 0,
    List<ValidatedMoodReference> validatedReferences =
        const <ValidatedMoodReference>[],
  }) async {
    final Stopwatch stopwatch = Stopwatch()..start();
    final File rawFile = File(imagePath);
    try {
      _lastError = null;
      final FaceDetector? detector = _faceDetector;
      if (detector == null) {
        _lastError = 'El detector facial no está inicializado.';
        return null;
      }

      // 1. Face Detection First
      final InputImage inputImage = InputImage.fromFilePath(imagePath);
      final List<Face> faces = await detector.processImage(inputImage);
      if (faces.isEmpty) {
        // Immediately delete the temporary image file and exit early
        await _deleteTempFileSafely(rawFile);
        return null;
      }

      final Interpreter? interpreter = _interpreter;
      if (interpreter == null) {
        _lastError = 'El intérprete TFLite no está cargado.';
        return null;
      }

      // 2. Decode & Crop Largest Face with 10% Padding
      final Uint8List bytes = await rawFile.readAsBytes();
      final img.Image? decodedImage = ImageUtils.decodeBytes(bytes);
      if (decodedImage == null) {
        _lastError = 'No se pudo decodificar la imagen capturada.';
        return null;
      }

      final Face primaryFace = _selectPrimaryFace(faces);
      final Rect box = primaryFace.boundingBox;
      final img.Image croppedFace = ImageUtils.cropFaceWithPadding(
        decodedImage,
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
      );

      // Extract facial position, pose angles, landmarks, and dense contours
      final FacialPositionDescriptor positionDescriptor =
          _extractPositionDescriptor(
        primaryFace,
        imageWidth: decodedImage.width,
        imageHeight: decodedImage.height,
      );
      final List<double> featureVector = ImageUtils.extractFacialFeatureVector(
        croppedFace,
        position: positionDescriptor,
        scanResolution: scanResolution,
      );

      // 3. Preprocess to normalized model input tensor (48x48x1 or 224x224x3)
      final List<int> inputShape = interpreter.getInputTensor(0).shape;
      final int targetSize = inputShape.length >= 3
          ? inputShape[1]
          : AppConstants.defaultInputSize;
      final int channels = inputShape.length >= 4 ? inputShape[3] : 1;

      final Uint8List processedPreviewBytes =
          ImageUtils.encodeProcessedPreviewPng(
        croppedFace,
        targetSize: processedImageSize,
        grayscale: channels == 1,
      );

      final List<List<List<List<double>>>> inputTensor = channels == 3
          ? ImageUtils.preprocessToRgbTensor(
              croppedFace,
              targetSize: targetSize,
              range: normalization,
            )
          : ImageUtils.preprocessToGrayscaleTensor(
              croppedFace,
              targetSize: targetSize,
              range: normalization,
            );

      // 4. Inference & Position + Reference Postprocessing
      final List<dynamic> output =
          List<double>.filled(1 * AppConstants.emotionLabels.length, 0.0)
              .reshape(<int>[1, AppConstants.emotionLabels.length]);

      interpreter.run(inputTensor, output);

      final List<double> rawScores =
          (output.first as List<dynamic>).cast<double>();
      final List<double> probabilitiesList =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: rawScores,
        featureVector: featureVector,
        validatedReferences: validatedReferences,
      );

      int maxIndex = 0;
      double maxScore = probabilitiesList[0];
      final Map<String, double> probabilities = <String, double>{};

      for (int i = 0; i < AppConstants.emotionLabels.length; i++) {
        final String label = AppConstants.emotionLabels[i];
        final double score = probabilitiesList[i];
        probabilities[label] = score;
        if (score > maxScore) {
          maxScore = score;
          maxIndex = i;
        }
      }

      stopwatch.stop();
      final int totalDurationMs =
          captureDurationMs + stopwatch.elapsedMilliseconds;

      return EmotionInferenceResult(
        label: AppConstants.emotionLabels[maxIndex],
        confidence: maxScore,
        probabilities: probabilities,
        timestamp: DateTime.now(),
        processedImageBytes: processedPreviewBytes,
        processedImageSize: processedImageSize,
        modelInputSize: targetSize,
        scanResolution: scanResolution,
        processingDurationMs: totalDurationMs,
        featureVector: featureVector,
        rawTfliteScores: rawScores,
      );
    } catch (e) {
      _lastError = 'ML inference pipeline error: $e';
      return null;
    } finally {
      // 5. Cleanup: Always delete temporary raw photo file from disk
      await _deleteTempFileSafely(rawFile);
    }
  }

  /// Processes a photo at [imagePath] (either captured from the active camera
  /// or selected from an existing photo) to build a [ConfiguredMoodProfile]
  /// for [emotion].
  ///
  /// Uses ML Kit FaceDetector to crop the face with 10% padding and extract
  /// 3D pose/contour cues; if the user selected an already tightly cropped
  /// face image where no full-frame bounding box is returned, falls back to
  /// scanning the full image at high resolution (`224x224`).
  Future<ConfiguredMoodProfile?> extractMoodProfileFromImageFile(
    String imagePath, {
    required String emotion,
    required String source,
    int previewSize = 96,
    bool deleteTempFile = true,
  }) async {
    final File rawFile = File(imagePath);
    try {
      _lastError = null;
      if (!isReady) {
        await initialize();
      }

      final Uint8List bytes = await rawFile.readAsBytes();
      final img.Image? decodedImage = ImageUtils.decodeBytes(bytes);
      if (decodedImage == null) {
        _lastError = 'No se pudo decodificar la imagen seleccionada.';
        return null;
      }

      img.Image croppedFace = decodedImage;
      FacialPositionDescriptor positionDescriptor =
          const FacialPositionDescriptor();

      final FaceDetector? detector = _faceDetector;
      if (detector != null) {
        try {
          final InputImage inputImage = InputImage.fromFilePath(imagePath);
          final List<Face> faces = await detector.processImage(inputImage);
          if (faces.isNotEmpty) {
            final Face primaryFace = _selectPrimaryFace(faces);
            final Rect box = primaryFace.boundingBox;
            croppedFace = ImageUtils.cropFaceWithPadding(
              decodedImage,
              left: box.left,
              top: box.top,
              width: box.width,
              height: box.height,
            );
            positionDescriptor = _extractPositionDescriptor(
              primaryFace,
              imageWidth: decodedImage.width,
              imageHeight: decodedImage.height,
            );
          }
        } catch (_) {
          // Fallback to full decoded image if ML Kit cannot run on the file
        }
      }

      final List<double> featureVector = ImageUtils.extractFacialFeatureVector(
        croppedFace,
        position: positionDescriptor,
        scanResolution: AppConstants.highResFacialScanSize,
      );

      final Uint8List previewBytes = ImageUtils.encodeProcessedPreviewPng(
        croppedFace,
        targetSize: previewSize,
        grayscale: true,
      );

      return ConfiguredMoodProfile(
        emotion: emotion,
        featureVector: featureVector,
        updatedAt: DateTime.now(),
        previewImageBytes: previewBytes,
        source: source,
      );
    } catch (e) {
      _lastError = 'Error al procesar imagen de referencia: $e';
      return null;
    } finally {
      if (deleteTempFile) {
        await _deleteTempFileSafely(rawFile);
      }
    }
  }

  FacialPositionDescriptor _extractPositionDescriptor(
    Face face, {
    required int imageWidth,
    required int imageHeight,
  }) {
    final Rect box = face.boundingBox;
    final double normCx =
        imageWidth > 0 ? (box.center.dx / imageWidth) - 0.5 : 0.0;
    final double normCy =
        imageHeight > 0 ? (box.center.dy / imageHeight) - 0.5 : 0.0;
    final double aspect =
        box.height > 0 ? (box.width / box.height).clamp(0.5, 1.4) : 0.85;

    double mouthSlope = 0.0;
    double mouthAperture = 0.0;
    double eyebrowArch = 0.0;
    double eyeAspect = 0.0;
    double lipContourOpening = 0.0;
    int totalContourPoints = 0;

    for (final FaceContour? contour in face.contours.values) {
      if (contour != null) {
        totalContourPoints += contour.points.length;
      }
    }

    final FaceLandmark? leftMouth = face.landmarks[FaceLandmarkType.leftMouth];
    final FaceLandmark? rightMouth =
        face.landmarks[FaceLandmarkType.rightMouth];
    final FaceLandmark? bottomMouth =
        face.landmarks[FaceLandmarkType.bottomMouth];
    final FaceLandmark? noseBase = face.landmarks[FaceLandmarkType.noseBase];

    if (leftMouth != null &&
        rightMouth != null &&
        bottomMouth != null &&
        box.height > 0) {
      final double cornersMeanY =
          (leftMouth.position.y + rightMouth.position.y) * 0.5;
      // Positive when corners are higher (smaller Y) than bottomMouth
      mouthSlope = ((bottomMouth.position.y - cornersMeanY) / box.height)
          .clamp(-0.5, 0.5);
    }

    if (noseBase != null && bottomMouth != null && box.height > 0) {
      final double noseToChin =
          (bottomMouth.position.y - noseBase.position.y) / box.height;
      mouthAperture = ((noseToChin - 0.22) * 3.0).clamp(-0.6, 0.8);
    }

    // Refine with dense facial contours when available
    if (box.height > 0) {
      final List<math.Point<int>>? upperLipBottom =
          face.contours[FaceContourType.upperLipBottom]?.points;
      final List<math.Point<int>>? lowerLipTop =
          face.contours[FaceContourType.lowerLipTop]?.points;

      if (upperLipBottom != null &&
          lowerLipTop != null &&
          upperLipBottom.length >= 3 &&
          lowerLipTop.length >= 3) {
        final math.Point<int> upperMid =
            upperLipBottom[upperLipBottom.length ~/ 2];
        final math.Point<int> lowerMid = lowerLipTop[lowerLipTop.length ~/ 2];
        final double innerGap = (lowerMid.y - upperMid.y) / box.height;
        lipContourOpening = ((innerGap - 0.025) * 6.0).clamp(-0.4, 1.0);

        final double leftCornerY =
            (upperLipBottom.first.y + lowerLipTop.last.y) * 0.5;
        final double rightCornerY =
            (upperLipBottom.last.y + lowerLipTop.first.y) * 0.5;
        final double cornersY = (leftCornerY + rightCornerY) * 0.5;
        final double centerLipY = (upperMid.y + lowerMid.y) * 0.5;
        final double contourSlope =
            ((centerLipY - cornersY) / box.height).clamp(-0.5, 0.5);
        mouthSlope = (0.45 * mouthSlope) + (0.55 * contourSlope);
      }

      final List<math.Point<int>>? leftEyePts =
          face.contours[FaceContourType.leftEye]?.points;
      final List<math.Point<int>>? rightEyePts =
          face.contours[FaceContourType.rightEye]?.points;
      double eyeCenterYSum = 0.0;
      int eyeCount = 0;

      double computeSingleEyeRatio(List<math.Point<int>> pts) {
        int minX = pts.first.x;
        int maxX = pts.first.x;
        int minY = pts.first.y;
        int maxY = pts.first.y;
        for (final math.Point<int> p in pts) {
          if (p.x < minX) minX = p.x;
          if (p.x > maxX) maxX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.y > maxY) maxY = p.y;
        }
        eyeCenterYSum += (minY + maxY) * 0.5;
        eyeCount++;
        final int w = math.max(1, maxX - minX);
        final int h = math.max(0, maxY - minY);
        return h / w;
      }

      double earSum = 0.0;
      int earCount = 0;
      if (leftEyePts != null && leftEyePts.length >= 6) {
        earSum += computeSingleEyeRatio(leftEyePts);
        earCount++;
      }
      if (rightEyePts != null && rightEyePts.length >= 6) {
        earSum += computeSingleEyeRatio(rightEyePts);
        earCount++;
      }
      if (earCount > 0) {
        final double meanEar = earSum / earCount;
        eyeAspect = ((meanEar - 0.28) * 3.5).clamp(-0.8, 0.8);
      }

      final List<math.Point<int>>? leftBrowPts =
          face.contours[FaceContourType.leftEyebrowTop]?.points;
      final List<math.Point<int>>? rightBrowPts =
          face.contours[FaceContourType.rightEyebrowTop]?.points;
      if (eyeCount > 0 &&
          ((leftBrowPts != null && leftBrowPts.isNotEmpty) ||
              (rightBrowPts != null && rightBrowPts.isNotEmpty))) {
        double browYSum = 0.0;
        int browPtsCount = 0;
        if (leftBrowPts != null) {
          for (final math.Point<int> p in leftBrowPts) {
            browYSum += p.y;
            browPtsCount++;
          }
        }
        if (rightBrowPts != null) {
          for (final math.Point<int> p in rightBrowPts) {
            browYSum += p.y;
            browPtsCount++;
          }
        }
        if (browPtsCount > 0) {
          final double meanEyeY = eyeCenterYSum / eyeCount;
          final double meanBrowY = browYSum / browPtsCount;
          final double browToEyeDist = (meanEyeY - meanBrowY) / box.height;
          eyebrowArch = ((browToEyeDist - 0.11) * 4.5).clamp(-0.8, 0.9);
        }
      }
    }

    return FacialPositionDescriptor(
      headPitchX: face.headEulerAngleX ?? 0.0,
      headYawY: face.headEulerAngleY ?? 0.0,
      headRollZ: face.headEulerAngleZ ?? 0.0,
      normalizedCenterX: normCx,
      normalizedCenterY: normCy,
      boxAspectRatio: aspect,
      smilingProbability: face.smilingProbability,
      leftEyeOpenProbability: face.leftEyeOpenProbability,
      rightEyeOpenProbability: face.rightEyeOpenProbability,
      mouthCornerSlope: mouthSlope,
      mouthApertureRatio: mouthAperture,
      eyebrowArchRatio: eyebrowArch,
      eyeAspectRatio: eyeAspect,
      lipContourAperture: lipContourOpening,
      contourPointsCount: totalContourPoints,
    );
  }

  Face _selectPrimaryFace(List<Face> faces) {
    Face largest = faces.first;
    double largestArea =
        math.max(0.0, largest.boundingBox.width * largest.boundingBox.height);
    for (final Face face in faces.skip(1)) {
      final double area = face.boundingBox.width * face.boundingBox.height;
      if (area > largestArea) {
        largest = face;
        largestArea = area;
      }
    }
    return largest;
  }

  Future<void> _deleteTempFileSafely(File file) async {
    try {
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {
      // Ignore file deletion errors in cleanup block
    }
  }

  /// Releases native ML Kit and TFLite interpreter handles.
  Future<void> dispose() async {
    try {
      await _faceDetector?.close();
      _interpreter?.close();
    } catch (e) {
      _lastError = 'Error disposing ML resources: $e';
    } finally {
      _faceDetector = null;
      _interpreter = null;
    }
  }
}
