# Flutter embedding rules are supplied automatically by the Flutter Gradle plugin.

# Play Core (deferred components / split install) is not used by this app.
# Flutter's embedding references it optionally, so suppress missing-class errors.
-dontwarn com.google.android.play.core.**

# TensorFlow Lite (CPU-only execution on Snapdragon 660; GPU delegate intentionally not bundled)
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# Google ML Kit Face Detection (Offline)
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_face.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_face_bundled.** { *; }
-dontwarn com.google.mlkit.**
