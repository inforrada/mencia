1. Project Overview & Role
You are an expert Mobile Software Engineer specializing in Flutter, Embedded Machine Learning (TFLite), and Android hardware constraints.
Your objective is to build a Flutter application that captures photos periodically from the front camera, detects faces offline using Google ML Kit, crops and preprocesses the face, and runs an offline TensorFlow Lite model to infer the emotional state of the user.
Target Device: Xiaomi Mi A2 (Snapdragon 660, Adreno 512, 4GB RAM, Android One 9/10).
2. Hard Architectural Rules & Constraints
Strict Offline Execution:
Under NO circumstances can images, embeddings, or inference data be sent to third-party cloud APIs.
All computer vision pipelines must execute locally on the Snapdragon 660 CPU/NPU.
Hardware & Thermal Preservation (Xiaomi Mi A2):
The Snapdragon 660 experiences thermal throttling if inferences run in tight loops.
Do NOT run inference on live camera streams (startImageStream). Capture isolated frames via scheduled timers (takePicture) to keep the SoC cool.
Restrict TFLite inference to 2 CPU worker threads (InterpreterOptions()..threads = 2). Avoid GPU delegates as driver overhead on Android 9/10 can cause instability.
Background Execution & Wakelock:
Use wakelock_plus to keep the screen active with minimal brightness while the tracker is running, preventing Android Doze mode from freezing timer interrupts.
3. Tech Stack & Dependencies
The project uses the following dependencies in pubspec.yaml:
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0+2
  google_mlkit_face_detection: ^0.12.0
  tflite_flutter: ^0.11.0
  image: ^4.2.0
  path_provider: ^2.1.2
  shared_preferences: ^2.2.3
  wakelock_plus: ^1.2.0
  intl: ^0.19.0


Android Build Requirements (android/app/build.gradle):
minSdkVersion: 21 (or 24 for optimal TFLite bindings)
targetSdkVersion: 33 or 34
aaptOptions { noCompress 'tflite' } must be present.
4. Directory & Module Structure
When generating or editing code, strictly adhere to the following architecture:
lib/
├── main.dart
├── core/
│   ├── constants/            # Model labels, default intervals, asset paths
│   └── utils/                # Image preprocessing, bitmap conversion, tensor resizing
├── services/
│   ├── camera_service.dart   # Front camera initialization, snapshot triggering
│   ├── ml_service.dart       # Google ML Kit FaceDetector & TFLite Interpreter
│   └── storage_service.dart  # Shared preferences & local JSON/SQLite mood history
└── presentation/
    ├── screens/
    │   ├── home_screen.dart       # Live status, last detected mood, counter
    │   ├── settings_screen.dart   # Interval selector (seconds/minutes), camera picker
    │   └── history_screen.dart    # Historical mood timeline/chart
    └── widgets/


5. Machine Learning Pipeline Invariants
When implementing ml_service.dart:
Face Detection First:
Initialize FaceDetector(options: FaceDetectorOptions(performanceMode: FaceDetectorMode.fast)).
Read image with InputImage.fromFilePath(imagePath).
If no face detected, immediately delete the temporary image file and exit early.
Preprocessing:
Decode image via package:image/image.dart.
Crop to bounding box with 10% padding around the face.
Resize to model input format (standard: 48x48 grayscale or 224x224 RGB depending on model).
Normalize pixel floats to [0.0, 1.0] or [-1.0, 1.0].
Inference & Postprocessing:
Run inference into buffer: var output = List.filled(1 * 7, 0.0).reshape([1, 7]).
Map highest softmax score to emotion classes: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'].
Cleanup:
Always delete the temporary raw photo file from disk after inference to avoid filling up the 64GB storage of the device.
6. Coding Standards & Agent Workflow
Error Handling: Every async operation interacting with camera hardware or model loading must be wrapped in try/catch blocks with explicit error states.
Verification Commands:
Check formatting: flutter format --set-exit-if-changed .
Analyze code: flutter analyze
Run unit tests: flutter test
