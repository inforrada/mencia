/// Core application constants, TFLite model metadata, and hardware constraints
/// tailored for Xiaomi Mi A2 (Snapdragon 660, Adreno 512, 4GB RAM, Android 9/10).
class AppConstants {
  AppConstants._();

  /// Application display title.
  static const String appName = 'Mencia Estado de Ánimo';

  /// Path to the bundled offline TensorFlow Lite model asset.
  static const String emotionModelAssetPath =
      'assets/models/emotion_model.tflite';

  /// Standard 7-class FER-2013 emotion labels in canonical output tensor index order.
  static const List<String> emotionLabels = <String>[
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral',
  ];

  /// Spanish (Spain) display names for each canonical emotion label.
  static const Map<String, String> emotionLabelsEs = <String, String>{
    'Angry': 'Enfado',
    'Disgust': 'Asco',
    'Fear': 'Miedo',
    'Happy': 'Alegría',
    'Sad': 'Tristeza',
    'Surprise': 'Sorpresa',
    'Neutral': 'Neutral',
  };

  /// Returns the Spanish (Spain) display label for a canonical emotion key.
  static String translateEmotion(String? emotion) {
    if (emotion == null || emotion.isEmpty) {
      return 'Ninguno';
    }
    return emotionLabelsEs[emotion] ?? emotion;
  }

  /// Number of CPU worker threads allocated to the TFLite interpreter.
  /// Restricted to 2 on Snapdragon 660 to prevent thermal throttling.
  static const int tfliteCpuThreads = 2;

  /// Bounding box fractional padding applied around detected faces (10%).
  static const double faceCropPaddingRatio = 0.10;

  /// Standard grayscale input spatial size (48x48x1).
  static const int grayscaleInputSize = 48;

  /// Standard RGB MobileNet/ViT input spatial size (224x224x3).
  static const int rgbInputSize = 224;

  /// High-resolution spatial grid size (224x224) used to scan facial positions,
  /// contours, and multi-scale Sobel gradients for higher accuracy.
  static const int highResFacialScanSize = 224;

  /// Default model input spatial dimensions.
  static const int defaultInputSize = grayscaleInputSize;

  /// Selectable resolutions (square side in pixels) for the treated face image.
  /// 48x48 is the smallest and matches the bundled model's native input.
  static const List<int> processedImageSizes = <int>[48, 64, 96, 128, 224];

  /// Default treated face image resolution.
  static const int defaultProcessedImageSize = 48;

  /// Minimum probability threshold (5%) to display secondary/blended moods.
  static const double minSecondaryMoodProbability = 0.05;

  /// Default periodic capture interval in seconds.
  static const int defaultCaptureIntervalSeconds = 30;

  /// Minimum allowed capture interval in seconds to preserve SoC thermal headroom.
  static const int minCaptureIntervalSeconds = 10;

  /// Maximum configurable capture interval in seconds (60 minutes).
  static const int maxCaptureIntervalSeconds = 3600;

  /// SharedPreferences keys for local offline persistence.
  static const String prefCaptureIntervalKey = 'capture_interval_seconds';
  static const String prefIntervalUnitMinutesKey = 'interval_unit_is_minutes';
  static const String prefSelectedCameraIdKey = 'selected_camera_id';
  static const String prefNormalizeSymmetricKey = 'normalize_minus_one_to_one';
  static const String prefLowBrightnessModeKey = 'low_brightness_wakelock_mode';
  static const String prefTrainingModeKey = 'training_mode_enabled';
  static const String prefProcessedImageSizeKey = 'processed_image_size_px';
  static const String prefMoodHistoryKey = 'mood_history_records_json';
  static const String prefConfiguredMoodProfilesKey =
      'configured_mood_profiles_json';

  /// Platform channel used to pick an existing image from device storage offline.
  static const String imagePickerChannelName =
      'com.example.mencia/image_picker';
}
