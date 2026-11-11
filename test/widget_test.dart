import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:mencia/core/constants/app_constants.dart';
import 'package:mencia/core/utils/image_utils.dart';
import 'package:mencia/presentation/screens/settings_screen.dart';
import 'package:mencia/presentation/widgets/mood_status_card.dart';
import 'package:mencia/presentation/widgets/training_review_card.dart';
import 'package:mencia/services/camera_service.dart';
import 'package:mencia/services/storage_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  group('AppConstants & ML invariants', () {
    test('defines canonical 7 emotion classes and 2 CPU threads', () {
      expect(
        AppConstants.emotionLabels,
        equals(<String>[
          'Angry',
          'Disgust',
          'Fear',
          'Happy',
          'Sad',
          'Surprise',
          'Neutral',
        ]),
      );
      expect(AppConstants.tfliteCpuThreads, equals(2));
      expect(AppConstants.faceCropPaddingRatio, closeTo(0.10, 1e-6));
      expect(AppConstants.minSecondaryMoodProbability, closeTo(0.05, 1e-6));
    });

    test('cropFaceWithPadding applies 10% padding and clamps to image bounds',
        () {
      final img.Image source = img.Image(width: 200, height: 200);
      final img.Image cropped = ImageUtils.cropFaceWithPadding(
        source,
        left: 50,
        top: 50,
        width: 100,
        height: 100,
      );
      expect(cropped.width, equals(120));
      expect(cropped.height, equals(120));
    });

    test(
        'preprocessToGrayscaleTensor produces [1, 48, 48, 1] normalized tensor',
        () {
      final img.Image sample = img.Image(width: 64, height: 64);
      final List<List<List<List<double>>>> tensor =
          ImageUtils.preprocessToGrayscaleTensor(sample);
      expect(tensor.length, equals(1));
      expect(tensor.first.length, equals(48));
      expect(tensor.first.first.length, equals(48));
      expect(tensor.first.first.first.length, equals(1));
    });

    test('encodeProcessedPreviewPng generates valid 48x48 PNG bytes', () {
      final img.Image sample = img.Image(width: 80, height: 80);
      final Uint8List pngBytes = ImageUtils.encodeProcessedPreviewPng(sample);
      expect(pngBytes, isNotEmpty);
      final img.Image? decoded = img.decodePng(pngBytes);
      expect(decoded, isNotNull);
      expect(decoded!.width, equals(48));
      expect(decoded.height, equals(48));
    });

    test('offers five treated image sizes with 48x48 as the smallest', () {
      expect(AppConstants.processedImageSizes.length, equals(5));
      expect(
        AppConstants.processedImageSizes
            .reduce((int a, int b) => a < b ? a : b),
        equals(48),
      );
      expect(
        AppConstants.processedImageSizes,
        contains(AppConstants.defaultProcessedImageSize),
      );
    });

    test('encodeProcessedPreviewPng honours every configurable size', () {
      final img.Image sample = img.Image(width: 300, height: 300);
      for (final int px in AppConstants.processedImageSizes) {
        final img.Image? decoded = img.decodePng(
          ImageUtils.encodeProcessedPreviewPng(sample, targetSize: px),
        );
        expect(decoded, isNotNull);
        expect(decoded!.width, equals(px));
        expect(decoded.height, equals(px));
      }
    });

    test('StorageService persists supported sizes and rejects others',
        () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final StorageService storage = StorageService();
      expect(await storage.getProcessedImageSize(), equals(48));

      await storage.setProcessedImageSize(128);
      expect(await storage.getProcessedImageSize(), equals(128));

      await storage.setProcessedImageSize(100);
      expect(storage.lastError, contains('Unsupported'));
      expect(await storage.getProcessedImageSize(), equals(128));
    });

    test('filterSignificantMoods returns secondary moods >= 5% sorted', () {
      final Map<String, double> probs = <String, double>{
        'Neutral': 0.45,
        'Happy': 0.32,
        'Surprise': 0.14,
        'Sad': 0.05,
        'Angry': 0.03,
        'Fear': 0.01,
        'Disgust': 0.00,
      };

      final List<MapEntry<String, double>> secondaries =
          ImageUtils.filterSignificantMoods(
        probs,
        excludeEmotion: 'Neutral',
      );

      expect(
        secondaries.map((MapEntry<String, double> e) => e.key).toList(),
        equals(<String>['Happy', 'Surprise', 'Sad']),
      );
    });

    test(
        'computePositionAndReferenceProbabilities differentiates moods by facial position and uses validated references',
        () {
      final img.Image sampleFace = img.Image(width: 48, height: 48);
      final List<double> uniformTflite = List<double>.filled(7, 1.0 / 7.0);

      // 1. Smiling facial position -> leans decisively toward Happy (index 3)
      final List<double> happyFeatures = ImageUtils.extractFacialFeatureVector(
        sampleFace,
        position: const FacialPositionDescriptor(
          headPitchX: 2.0,
          smilingProbability: 0.90,
          mouthCornerSlope: 0.25,
          boxAspectRatio: 0.95,
        ),
      );
      final List<double> happyProbs =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: uniformTflite,
        featureVector: happyFeatures,
      );
      expect(happyProbs[3], greaterThan(0.40)); // Happy > 40% (not 14.3%)

      // 2. Downward head pitch + drooped mouth + low eye openness -> leans toward Sad (index 4)
      final List<double> sadFeatures = ImageUtils.extractFacialFeatureVector(
        sampleFace,
        position: const FacialPositionDescriptor(
          headPitchX: -18.0,
          smilingProbability: 0.05,
          leftEyeOpenProbability: 0.25,
          rightEyeOpenProbability: 0.25,
          mouthCornerSlope: -0.25,
        ),
      );
      final List<double> sadProbs =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: uniformTflite,
        featureVector: sadFeatures,
      );
      expect(sadProbs[4], greaterThan(0.35)); // Sad dominates

      // 3. Validated reference mood pulls a similar feature vector toward the confirmed mood
      final List<double> probsWithAngryReference =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: uniformTflite,
        featureVector: sadFeatures,
        validatedReferences: <ValidatedMoodReference>[
          ValidatedMoodReference(
            emotion: 'Angry',
            featureVector: sadFeatures,
          ),
        ],
      );
      // Angry (index 0) is significantly boosted by the validated reference
      expect(probsWithAngryReference[0], greaterThan(sadProbs[0] + 0.40));
    });

    test(
        'MoodRecord serializes training fields, secondary probabilities, featureVector, and processingDurationMs',
        () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final StorageService storage = StorageService();
      final DateTime now = DateTime.utc(2026, 11, 10, 17, 0, 0);
      final MoodRecord record = MoodRecord(
        emotion: 'Happy',
        predictedEmotion: 'Neutral',
        confidence: 0.48,
        timestamp: now,
        isTrainingVerified: true,
        wasCorrected: true,
        probabilities: const <String, double>{
          'Neutral': 0.48,
          'Happy': 0.35,
          'Surprise': 0.10,
          'Angry': 0.03,
        },
        featureVector: const <double>[0.1, -0.2, 0.5, 0.8],
        processingDurationMs: 1420,
      );
      await storage.appendMoodRecord(record);

      final List<MoodRecord> history = await storage.getMoodHistory();
      expect(history.first.processingDurationMs, equals(1420));
      expect(
        ImageUtils.formatProcessingDuration(1420),
        equals('1.42 s (1420 ms)'),
      );

      final List<ValidatedMoodReference> refs =
          await storage.getValidatedMoodReferences();
      expect(refs.length, equals(1));
      expect(refs.first.emotion, equals('Happy'));
      expect(refs.first.featureVector.length, equals(4));
    });

    testWidgets(
        'MoodStatusCard and TrainingReviewCard display secondary moods >= 5% and processing duration',
        (WidgetTester tester) async {
      final img.Image sampleImg = img.Image(width: 48, height: 48);
      final Uint8List previewBytes =
          ImageUtils.encodeProcessedPreviewPng(sampleImg);
      const Map<String, double> probabilities = <String, double>{
        'Neutral': 0.46,
        'Happy': 0.34,
        'Surprise': 0.12,
        'Angry': 0.04,
      };
      final TrainingSample sample = TrainingSample(
        id: '1',
        predictedEmotion: 'Neutral',
        confidence: 0.46,
        timestamp: DateTime(2026, 11, 10, 19, 30),
        probabilities: probabilities,
        processedImageBytes: previewBytes,
        processingDurationMs: 1850,
      );

      String? correctedTo;
      bool confirmed = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: Column(
                children: <Widget>[
                  const MoodStatusCard(
                    isTracking: true,
                    lastEmotion: 'Neutral',
                    lastConfidence: 0.46,
                    lastProbabilities: probabilities,
                    lastProcessingDurationMs: 1850,
                    inferenceCount: 3,
                    validatedReferenceCount: 2,
                  ),
                  TrainingReviewCard(
                    samples: <TrainingSample>[sample],
                    onConfirmSample: (TrainingSample s) async {
                      confirmed = true;
                    },
                    onCorrectSample: (TrainingSample s, String emotion) async {
                      correctedTo = emotion;
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      );

      expect(find.text('Estado de ánimo decidido:'), findsOneWidget);
      expect(find.text('Neutral'), findsOneWidget);
      expect(find.text('Alegría: 34.0%'), findsNWidgets(2));
      expect(find.text('Sorpresa: 12.0%'), findsNWidgets(2));
      expect(find.text('Enfado: 4.0%'), findsNothing);
      expect(
        find.text('Tiempo de procesado: 1.85 s (1850 ms)'),
        findsNWidgets(2),
      );

      await tester.ensureVisible(find.text('Rechazar'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Rechazar'));
      await tester.pumpAndSettle();

      expect(
        find.text('Indica cuál es el estado de ánimo correcto:'),
        findsOneWidget,
      );
      expect(find.text('Guardar estado de ánimo correcto'), findsOneWidget);

      await tester.ensureVisible(find.text('Guardar estado de ánimo correcto'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Guardar estado de ánimo correcto'));
      await tester.pumpAndSettle();

      expect(correctedTo, equals('Happy'));
      expect(confirmed, isFalse);
    });

    test(
        'ConfiguredMoodProfile persists per emotion and acts as weighted reference for inference',
        () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final StorageService storage = StorageService();
      final img.Image sample = img.Image(width: 64, height: 64);
      final Uint8List previewBytes =
          ImageUtils.encodeProcessedPreviewPng(sample);
      final List<double> features = ImageUtils.extractFacialFeatureVector(
        sample,
        position: const FacialPositionDescriptor(
          headPitchX: 12.0,
          mouthApertureRatio: 0.45,
        ),
      );

      final ConfiguredMoodProfile profile = ConfiguredMoodProfile(
        emotion: 'Surprise',
        featureVector: features,
        updatedAt: DateTime.utc(2026, 11, 11, 11, 15),
        previewImageBytes: previewBytes,
        source: 'existing',
      );
      await storage.setConfiguredMoodProfile(profile);

      final Map<String, ConfiguredMoodProfile> loaded =
          await storage.getConfiguredMoodProfiles();
      expect(loaded.containsKey('Surprise'), isTrue);
      expect(loaded['Surprise']!.source, equals('existing'));
      expect(loaded['Surprise']!.previewImageBytes, isNotEmpty);

      final List<ValidatedMoodReference> refs =
          await storage.getValidatedMoodReferences();
      expect(refs.length, equals(1));
      expect(refs.first.emotion, equals('Surprise'));
      expect(refs.first.weight, closeTo(2.5, 1e-6));

      final List<double> probs =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: List<double>.filled(7, 1.0 / 7.0),
        featureVector: features,
        validatedReferences: refs,
      );
      // Surprise (index 5) must be the top emotion established by the reference
      expect(probs[5], greaterThan(0.55));
    });

    testWidgets(
        'SettingsScreen displays mood definition section and assigns existing photo reference',
        (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final StorageService storage = StorageService();
      final CameraService camera = CameraService();
      final img.Image sampleImg = img.Image(width: 48, height: 48);
      final Uint8List previewBytes =
          ImageUtils.encodeProcessedPreviewPng(sampleImg);
      final List<double> features =
          ImageUtils.extractFacialFeatureVector(sampleImg);

      final TrainingSample existingSample = TrainingSample(
        id: 'sample-1',
        predictedEmotion: 'Happy',
        confidence: 0.80,
        timestamp: DateTime(2026, 11, 11, 11, 0),
        processedImageBytes: previewBytes,
        featureVector: features,
      );

      await tester.pumpWidget(
        MaterialApp(
          home: SettingsScreen(
            storageService: storage,
            cameraService: camera,
            existingSamples: <TrainingSample>[existingSample],
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(
        find.text('Definición de estados de ánimo de referencia'),
        findsOneWidget,
      );
      expect(find.text('0/7 definidos'), findsOneWidget);

      final Finder angryExistingBtn =
          find.byKey(const ValueKey<String>('mood_existing_btn_Angry'));
      await tester.ensureVisible(angryExistingBtn);
      await tester.pumpAndSettle();
      await tester.tap(angryExistingBtn);
      await tester.pumpAndSettle();

      expect(
        find.text('Seleccionar foto existente para Enfado'),
        findsOneWidget,
      );

      // Tap the existing session capture thumbnail to assign it to Angry
      await tester.tap(find.text('Alegría').last);
      await tester.pumpAndSettle();

      expect(find.text('1/7 definidos'), findsOneWidget);
      expect(
        find.text('Referencia activa (Foto existente)'),
        findsOneWidget,
      );
    });
  });
}
