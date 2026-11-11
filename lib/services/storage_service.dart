import 'dart:convert';
import 'dart:typed_data';
import 'package:shared_preferences/shared_preferences.dart';
import '../core/constants/app_constants.dart';
import '../core/utils/image_utils.dart';

/// Represents a user-defined reference image and extracted facial feature
/// vector for one of the 7 emotion classes, configured in Settings either by
/// selecting an existing photo or capturing one with the active camera.
class ConfiguredMoodProfile {
  const ConfiguredMoodProfile({
    required this.emotion,
    required this.featureVector,
    required this.updatedAt,
    this.previewImageBytes,
    this.source = 'camera',
  });

  factory ConfiguredMoodProfile.fromJson(Map<String, dynamic> json) {
    final List<double> parsedFeatures = <double>[];
    final Object? rawFeatures = json['featureVector'];
    if (rawFeatures is List<dynamic>) {
      for (final dynamic item in rawFeatures) {
        if (item is num) {
          parsedFeatures.add(item.toDouble());
        }
      }
    }
    Uint8List? previewBytes;
    final String? b64 = json['previewPngBase64'] as String?;
    if (b64 != null && b64.isNotEmpty) {
      try {
        previewBytes = base64Decode(b64);
      } catch (_) {}
    }
    return ConfiguredMoodProfile(
      emotion: json['emotion'] as String? ?? 'Neutral',
      featureVector: parsedFeatures,
      updatedAt: DateTime.tryParse(json['updatedAt'] as String? ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
      previewImageBytes: previewBytes,
      source: json['source'] as String? ?? 'camera',
    );
  }

  /// Target emotion class label (`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`).
  final String emotion;

  /// Extracted 14-D high-resolution facial position & appearance feature vector.
  final List<double> featureVector;

  /// Timestamp when this reference was defined or updated.
  final DateTime updatedAt;

  /// Cropped and processed face thumbnail encoded as PNG for visual preview in Settings.
  final Uint8List? previewImageBytes;

  /// Origin of the reference photo (`'camera'` or `'existing'`).
  final String source;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'emotion': emotion,
        'featureVector': featureVector,
        'updatedAt': updatedAt.toIso8601String(),
        'source': source,
        if (previewImageBytes != null && previewImageBytes!.isNotEmpty)
          'previewPngBase64': base64Encode(previewImageBytes!),
      };
}

/// Serializable record representing a single local emotion inference entry,
/// including optional Training Mode validation metadata and facial feature
/// vector used as a reference for subsequent calculations.
class MoodRecord {
  const MoodRecord({
    required this.emotion,
    required this.confidence,
    required this.timestamp,
    this.predictedEmotion,
    this.isTrainingVerified = false,
    this.wasCorrected = false,
    this.probabilities = const <String, double>{},
    this.featureVector = const <double>[],
    this.processingDurationMs,
  });

  factory MoodRecord.fromJson(Map<String, dynamic> json) {
    final String emotion = json['emotion'] as String? ?? 'Neutral';
    final Map<String, double> parsedProbabilities = <String, double>{};
    final Object? rawProbs = json['probabilities'];
    if (rawProbs is Map<String, dynamic>) {
      rawProbs.forEach((String key, dynamic value) {
        if (value is num) {
          parsedProbabilities[key] = value.toDouble();
        }
      });
    }
    final List<double> parsedFeatures = <double>[];
    final Object? rawFeatures = json['featureVector'];
    if (rawFeatures is List<dynamic>) {
      for (final dynamic item in rawFeatures) {
        if (item is num) {
          parsedFeatures.add(item.toDouble());
        }
      }
    }
    return MoodRecord(
      emotion: emotion,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      timestamp: DateTime.tryParse(json['timestamp'] as String? ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
      predictedEmotion: json['predictedEmotion'] as String? ?? emotion,
      isTrainingVerified: json['isTrainingVerified'] as bool? ?? false,
      wasCorrected: json['wasCorrected'] as bool? ?? false,
      probabilities: parsedProbabilities,
      featureVector: parsedFeatures,
      processingDurationMs: (json['processingDurationMs'] as num?)?.toInt(),
    );
  }

  /// Final ground-truth or inferred emotion label.
  final String emotion;

  /// Model softmax confidence score `[0.0, 1.0]`.
  final double confidence;

  /// Capture timestamp.
  final DateTime timestamp;

  /// Original emotion predicted by the TFLite model before any user correction.
  final String? predictedEmotion;

  /// Whether this record was explicitly confirmed or corrected in Training Mode.
  final bool isTrainingVerified;

  /// Whether the user rejected the model's prediction and chose a different label.
  final bool wasCorrected;

  /// Full softmax probability distribution across the 7 emotion classes.
  final Map<String, double> probabilities;

  /// 14-dimensional facial position & spatial appearance feature vector.
  final List<double> featureVector;

  /// Total time in milliseconds spent capturing and processing the image.
  final int? processingDurationMs;

  /// Secondary moods with at least [minProbability] (default 5%), excluding [emotion].
  List<MapEntry<String, double>> secondaryMoods({
    double minProbability = AppConstants.minSecondaryMoodProbability,
  }) {
    return ImageUtils.filterSignificantMoods(
      probabilities,
      minProbability: minProbability,
      excludeEmotion: emotion,
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
        'emotion': emotion,
        'confidence': confidence,
        'timestamp': timestamp.toIso8601String(),
        'predictedEmotion': predictedEmotion ?? emotion,
        'isTrainingVerified': isTrainingVerified,
        'wasCorrected': wasCorrected,
        'probabilities': probabilities,
        if (featureVector.isNotEmpty) 'featureVector': featureVector,
        if (processingDurationMs != null)
          'processingDurationMs': processingDurationMs,
      };
}

/// Offline storage service persisting user settings and historical mood records
/// locally on-device using [SharedPreferences].
class StorageService {
  SharedPreferences? _prefs;
  String? _lastError;

  String? get lastError => _lastError;

  Future<SharedPreferences> _instance() async {
    return _prefs ??= await SharedPreferences.getInstance();
  }

  /// Reads the configured periodic capture interval in seconds.
  Future<int> getCaptureIntervalSeconds() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      return prefs.getInt(AppConstants.prefCaptureIntervalKey) ??
          AppConstants.defaultCaptureIntervalSeconds;
    } catch (e) {
      _lastError = 'Failed to read capture interval: $e';
      return AppConstants.defaultCaptureIntervalSeconds;
    }
  }

  /// Saves the periodic capture interval in seconds.
  Future<void> setCaptureIntervalSeconds(int seconds) async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      final int clamped = seconds.clamp(
        AppConstants.minCaptureIntervalSeconds,
        AppConstants.maxCaptureIntervalSeconds,
      );
      await prefs.setInt(AppConstants.prefCaptureIntervalKey, clamped);
    } catch (e) {
      _lastError = 'Failed to save capture interval: $e';
    }
  }

  /// Reads whether the interval selector UI is displaying minutes (`true`) or seconds (`false`).
  Future<bool> getIntervalUnitIsMinutes() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      return prefs.getBool(AppConstants.prefIntervalUnitMinutesKey) ?? false;
    } catch (e) {
      _lastError = 'Failed to read interval unit preference: $e';
      return false;
    }
  }

  /// Saves whether the interval selector UI is displaying minutes (`true`) or seconds (`false`).
  Future<void> setIntervalUnitIsMinutes(bool isMinutes) async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.setBool(AppConstants.prefIntervalUnitMinutesKey, isMinutes);
    } catch (e) {
      _lastError = 'Failed to save interval unit preference: $e';
    }
  }

  /// Reads the preferred camera identifier, if configured.
  Future<String?> getSelectedCameraId() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      return prefs.getString(AppConstants.prefSelectedCameraIdKey);
    } catch (e) {
      _lastError = 'Failed to read selected camera: $e';
      return null;
    }
  }

  /// Saves the preferred camera identifier.
  Future<void> setSelectedCameraId(String cameraId) async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.setString(AppConstants.prefSelectedCameraIdKey, cameraId);
    } catch (e) {
      _lastError = 'Failed to save selected camera: $e';
    }
  }

  /// Reads the configured tensor pixel normalization range.
  Future<NormalizationRange> getNormalizationRange() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      final bool symmetric =
          prefs.getBool(AppConstants.prefNormalizeSymmetricKey) ?? false;
      return symmetric
          ? NormalizationRange.minusOneToOne
          : NormalizationRange.zeroToOne;
    } catch (e) {
      _lastError = 'Failed to read normalization range: $e';
      return NormalizationRange.zeroToOne;
    }
  }

  /// Saves the tensor pixel normalization range preference.
  Future<void> setNormalizationRange(NormalizationRange range) async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.setBool(
        AppConstants.prefNormalizeSymmetricKey,
        range == NormalizationRange.minusOneToOne,
      );
    } catch (e) {
      _lastError = 'Failed to save normalization range: $e';
    }
  }

  /// Reads the configured treated face image resolution (square side in px).
  /// Falls back to the default if the stored value is not a supported size.
  Future<int> getProcessedImageSize() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      final int? stored = prefs.getInt(AppConstants.prefProcessedImageSizeKey);
      if (stored != null && AppConstants.processedImageSizes.contains(stored)) {
        return stored;
      }
      return AppConstants.defaultProcessedImageSize;
    } catch (e) {
      _lastError = 'Failed to read processed image size: $e';
      return AppConstants.defaultProcessedImageSize;
    }
  }

  /// Saves the treated face image resolution. Unsupported sizes are ignored.
  Future<void> setProcessedImageSize(int size) async {
    if (!AppConstants.processedImageSizes.contains(size)) {
      _lastError = 'Unsupported processed image size: $size';
      return;
    }
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.setInt(AppConstants.prefProcessedImageSizeKey, size);
    } catch (e) {
      _lastError = 'Failed to save processed image size: $e';
    }
  }

  /// Reads whether Training Mode (Modo Entrenamiento) is enabled.
  Future<bool> getTrainingModeEnabled() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      return prefs.getBool(AppConstants.prefTrainingModeKey) ?? false;
    } catch (e) {
      _lastError = 'Failed to read training mode state: $e';
      return false;
    }
  }

  /// Saves whether Training Mode (Modo Entrenamiento) is enabled.
  Future<void> setTrainingModeEnabled(bool enabled) async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.setBool(AppConstants.prefTrainingModeKey, enabled);
    } catch (e) {
      _lastError = 'Failed to save training mode state: $e';
    }
  }

  /// Loads stored mood history records in reverse chronological order.
  Future<List<MoodRecord>> getMoodHistory() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      final String? rawJson = prefs.getString(AppConstants.prefMoodHistoryKey);
      if (rawJson == null || rawJson.isEmpty) {
        return <MoodRecord>[];
      }
      final List<dynamic> decoded = jsonDecode(rawJson) as List<dynamic>;
      return decoded
          .whereType<Map<String, dynamic>>()
          .map(MoodRecord.fromJson)
          .toList();
    } catch (e) {
      _lastError = 'Failed to load mood history: $e';
      return <MoodRecord>[];
    }
  }

  /// Loads the map of user-configured Mood reference profiles (`emotion -> ConfiguredMoodProfile`).
  Future<Map<String, ConfiguredMoodProfile>> getConfiguredMoodProfiles() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      final String? rawJson =
          prefs.getString(AppConstants.prefConfiguredMoodProfilesKey);
      if (rawJson == null || rawJson.isEmpty) {
        return <String, ConfiguredMoodProfile>{};
      }
      final Map<String, dynamic> decoded =
          jsonDecode(rawJson) as Map<String, dynamic>;
      final Map<String, ConfiguredMoodProfile> result =
          <String, ConfiguredMoodProfile>{};
      decoded.forEach((String key, dynamic value) {
        if (value is Map<String, dynamic>) {
          final ConfiguredMoodProfile profile =
              ConfiguredMoodProfile.fromJson(value);
          if (profile.featureVector.isNotEmpty) {
            result[key] = profile;
          }
        }
      });
      return result;
    } catch (e) {
      _lastError = 'Failed to load configured mood profiles: $e';
      return <String, ConfiguredMoodProfile>{};
    }
  }

  /// Saves or updates the reference photo & feature vector for [profile.emotion].
  Future<void> setConfiguredMoodProfile(ConfiguredMoodProfile profile) async {
    try {
      _lastError = null;
      final Map<String, ConfiguredMoodProfile> current =
          await getConfiguredMoodProfiles();
      current[profile.emotion] = profile;
      final SharedPreferences prefs = await _instance();
      final Map<String, dynamic> serializable = <String, dynamic>{};
      current.forEach((String key, ConfiguredMoodProfile value) {
        serializable[key] = value.toJson();
      });
      await prefs.setString(
        AppConstants.prefConfiguredMoodProfilesKey,
        jsonEncode(serializable),
      );
    } catch (e) {
      _lastError = 'Failed to save configured mood profile: $e';
    }
  }

  /// Removes the configured reference photo for [emotion].
  Future<void> removeConfiguredMoodProfile(String emotion) async {
    try {
      _lastError = null;
      final Map<String, ConfiguredMoodProfile> current =
          await getConfiguredMoodProfiles();
      current.remove(emotion);
      final SharedPreferences prefs = await _instance();
      final Map<String, dynamic> serializable = <String, dynamic>{};
      current.forEach((String key, ConfiguredMoodProfile value) {
        serializable[key] = value.toJson();
      });
      await prefs.setString(
        AppConstants.prefConfiguredMoodProfilesKey,
        jsonEncode(serializable),
      );
    } catch (e) {
      _lastError = 'Failed to remove configured mood profile: $e';
    }
  }

  /// Returns all active reference prototypes for emotion inference, combining:
  /// 1. Explicitly configured Mood reference images from Settings (`weight: 2.5`),
  /// 2. Training-validated mood records from history (`weight: 1.0`).
  Future<List<ValidatedMoodReference>> getValidatedMoodReferences() async {
    final Map<String, ConfiguredMoodProfile> configured =
        await getConfiguredMoodProfiles();
    final List<MoodRecord> history = await getMoodHistory();

    final List<ValidatedMoodReference> references = <ValidatedMoodReference>[
      ...configured.values
          .where((ConfiguredMoodProfile p) => p.featureVector.isNotEmpty)
          .map(
            (ConfiguredMoodProfile p) => ValidatedMoodReference(
              emotion: p.emotion,
              featureVector: p.featureVector,
              weight: 2.5,
            ),
          ),
      ...history
          .where(
            (MoodRecord r) =>
                r.isTrainingVerified && r.featureVector.isNotEmpty,
          )
          .map(
            (MoodRecord r) => ValidatedMoodReference(
              emotion: r.emotion,
              featureVector: r.featureVector,
              weight: 1.0,
            ),
          ),
    ];
    return references;
  }

  /// Appends a new [MoodRecord] to local storage.
  Future<void> appendMoodRecord(MoodRecord record) async {
    try {
      _lastError = null;
      final List<MoodRecord> current = await getMoodHistory();
      final List<MoodRecord> updated = <MoodRecord>[record, ...current];
      final SharedPreferences prefs = await _instance();
      final String encoded =
          jsonEncode(updated.map((MoodRecord e) => e.toJson()).toList());
      await prefs.setString(AppConstants.prefMoodHistoryKey, encoded);
    } catch (e) {
      _lastError = 'Failed to append mood record: $e';
    }
  }

  /// Clears all locally stored mood history records.
  Future<void> clearMoodHistory() async {
    try {
      _lastError = null;
      final SharedPreferences prefs = await _instance();
      await prefs.remove(AppConstants.prefMoodHistoryKey);
    } catch (e) {
      _lastError = 'Failed to clear mood history: $e';
    }
  }
}
