import 'package:camera/camera.dart';
import 'package:camera/camera.dart' as camera_plugin;

/// Service responsible for managing the front camera lifecycle and capturing
/// isolated still snapshots on scheduled timers.
///
/// Architectural Rule: Never use `startImageStream` to avoid continuous SoC load
/// and thermal throttling on the Xiaomi Mi A2 (Snapdragon 660).
class CameraService {
  CameraController? _controller;
  List<CameraDescription> _availableCameras = const <CameraDescription>[];
  String? _lastError;

  CameraController? get controller => _controller;
  List<CameraDescription> get availableCameras => _availableCameras;
  String? get lastError => _lastError;
  bool get isInitialized => _controller?.value.isInitialized ?? false;
  CameraDescription? get activeCamera => _controller?.description;

  /// Queries available hardware cameras without opening a capture session.
  Future<List<CameraDescription>> discoverCameras() async {
    try {
      _lastError = null;
      _availableCameras = await camera_plugin.availableCameras();
      return _availableCameras;
    } catch (e) {
      _lastError = 'No se pudieron enumerar las cámaras: $e';
      return const <CameraDescription>[];
    }
  }

  /// Discovers device cameras and initializes the front-facing camera
  /// (or the camera matching [preferredCameraName] if provided).
  Future<bool> initialize({String? preferredCameraName}) async {
    try {
      _lastError = null;
      _availableCameras = await camera_plugin.availableCameras();
      if (_availableCameras.isEmpty) {
        _lastError = 'No se han detectado cámaras en el dispositivo.';
        return false;
      }

      final CameraDescription selectedCamera = _selectCamera(
        preferredCameraName: preferredCameraName,
      );

      await _controller?.dispose();
      CameraController controller = CameraController(
        selectedCamera,
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      try {
        await controller.initialize();
      } catch (_) {
        await controller.dispose();
        controller = CameraController(
          selectedCamera,
          ResolutionPreset.high,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.jpeg,
        );
        await controller.initialize();
      }
      await controller.setFlashMode(FlashMode.off);
      _controller = controller;
      return true;
    } catch (e) {
      _lastError = 'Error al inicializar la cámara: $e';
      return false;
    }
  }

  /// Captures a single isolated still frame and returns its temporary file path.
  Future<String?> takePeriodicSnapshot() async {
    final CameraController? activeController = _controller;
    if (activeController == null || !activeController.value.isInitialized) {
      _lastError = 'El controlador de la cámara no está inicializado.';
      return null;
    }
    if (activeController.value.isTakingPicture) {
      return null;
    }

    try {
      _lastError = null;
      final XFile file = await activeController.takePicture();
      return file.path;
    } catch (e) {
      _lastError = 'Error al tomar la captura: $e';
      return null;
    }
  }

  CameraDescription _selectCamera({String? preferredCameraName}) {
    if (preferredCameraName != null && preferredCameraName.isNotEmpty) {
      for (final CameraDescription camera in _availableCameras) {
        if (camera.name == preferredCameraName) {
          return camera;
        }
      }
    }
    for (final CameraDescription camera in _availableCameras) {
      if (camera.lensDirection == CameraLensDirection.front) {
        return camera;
      }
    }
    return _availableCameras.first;
  }

  /// Releases camera hardware resources.
  Future<void> dispose() async {
    try {
      await _controller?.dispose();
    } catch (e) {
      _lastError = 'Error disposing camera: $e';
    } finally {
      _controller = null;
    }
  }
}
