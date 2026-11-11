import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import '../../core/constants/app_constants.dart';
import '../../core/utils/image_utils.dart';
import '../../services/camera_service.dart';
import '../../services/ml_service.dart';
import '../../services/storage_service.dart';
import '../widgets/training_review_card.dart';

/// Screen for configuring periodic snapshot intervals (seconds/minutes),
/// selecting the target camera sensor, enabling Training Mode, choosing
/// pixel normalization, and defining reference photos for each of the 7 Moods
/// (either from an existing photo or captured with the active camera).
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    super.key,
    required this.storageService,
    required this.cameraService,
    this.mlService,
    this.existingSamples = const <TrainingSample>[],
  });

  final StorageService storageService;
  final CameraService cameraService;
  final MlService? mlService;
  final List<TrainingSample> existingSamples;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  static const MethodChannel _imagePickerChannel = MethodChannel(
    AppConstants.imagePickerChannelName,
  );

  int _intervalSeconds = AppConstants.defaultCaptureIntervalSeconds;
  bool _useMinutesUnit = false;
  bool _trainingModeEnabled = false;
  int _processedImageSize = AppConstants.defaultProcessedImageSize;
  String? _selectedCameraId;
  NormalizationRange _normalization = NormalizationRange.zeroToOne;
  List<CameraDescription> _cameras = const <CameraDescription>[];
  Map<String, ConfiguredMoodProfile> _configuredMoods =
      <String, ConfiguredMoodProfile>{};
  String? _busyEmotion;
  String? _errorMessage;

  static const List<int> _secondsOptions = <int>[10, 15, 30, 45, 60, 90, 120];
  static const List<int> _minutesOptions = <int>[1, 2, 5, 10, 15, 30, 60];

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final int interval =
        await widget.storageService.getCaptureIntervalSeconds();
    final bool useMinutes =
        await widget.storageService.getIntervalUnitIsMinutes();
    final bool trainingMode =
        await widget.storageService.getTrainingModeEnabled();
    final int imageSize = await widget.storageService.getProcessedImageSize();
    final String? savedCamera =
        await widget.storageService.getSelectedCameraId();
    final NormalizationRange norm =
        await widget.storageService.getNormalizationRange();
    final Map<String, ConfiguredMoodProfile> configuredMoods =
        await widget.storageService.getConfiguredMoodProfiles();
    final List<CameraDescription> discovered =
        await widget.cameraService.discoverCameras();

    if (!mounted) {
      return;
    }

    String? effectiveCamera = savedCamera;
    if (effectiveCamera == null && discovered.isNotEmpty) {
      final CameraDescription front = discovered.firstWhere(
        (CameraDescription c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => discovered.first,
      );
      effectiveCamera = front.name;
    }

    setState(() {
      _intervalSeconds = interval;
      _useMinutesUnit = useMinutes;
      _trainingModeEnabled = trainingMode;
      _processedImageSize = imageSize;
      _selectedCameraId = effectiveCamera;
      _normalization = norm;
      _configuredMoods = configuredMoods;
      _cameras = discovered;
      _errorMessage = widget.cameraService.lastError;
    });
  }

  Future<void> _toggleTrainingMode(bool enabled) async {
    await widget.storageService.setTrainingModeEnabled(enabled);
    if (!mounted) {
      return;
    }
    setState(() {
      _trainingModeEnabled = enabled;
    });
  }

  Future<void> _updateProcessedImageSize(int? size) async {
    if (size == null) {
      return;
    }
    await widget.storageService.setProcessedImageSize(size);
    if (!mounted) {
      return;
    }
    setState(() {
      _processedImageSize = size;
    });
  }

  Future<void> _toggleUnit(bool useMinutes) async {
    await widget.storageService.setIntervalUnitIsMinutes(useMinutes);
    int updatedSeconds = _intervalSeconds;
    if (useMinutes && updatedSeconds < 60) {
      updatedSeconds = 60;
      await widget.storageService.setCaptureIntervalSeconds(updatedSeconds);
    }
    if (!mounted) {
      return;
    }
    setState(() {
      _useMinutesUnit = useMinutes;
      _intervalSeconds = updatedSeconds;
    });
  }

  Future<void> _updateIntervalSeconds(int seconds) async {
    await widget.storageService.setCaptureIntervalSeconds(seconds);
    if (!mounted) {
      return;
    }
    setState(() {
      _intervalSeconds = seconds;
    });
  }

  Future<void> _updateCamera(String? cameraId) async {
    if (cameraId == null) {
      return;
    }
    await widget.storageService.setSelectedCameraId(cameraId);
    if (!mounted) {
      return;
    }
    setState(() {
      _selectedCameraId = cameraId;
    });
  }

  Future<void> _updateNormalization(NormalizationRange? range) async {
    if (range == null) {
      return;
    }
    await widget.storageService.setNormalizationRange(range);
    if (!mounted) {
      return;
    }
    setState(() {
      _normalization = range;
    });
  }

  Future<ConfiguredMoodProfile?> _buildProfileFromImageFile(
    String imagePath, {
    required String emotion,
    required String source,
  }) async {
    final MlService? ml = widget.mlService;
    if (ml != null) {
      return ml.extractMoodProfileFromImageFile(
        imagePath,
        emotion: emotion,
        source: source,
        previewSize: _processedImageSize,
      );
    }

    // Fallback when MlService is not injected (e.g. standalone widget tests)
    final File rawFile = File(imagePath);
    try {
      final Uint8List bytes = await rawFile.readAsBytes();
      final img.Image? decoded = ImageUtils.decodeBytes(bytes);
      if (decoded == null) {
        _errorMessage = 'No se pudo decodificar la imagen de referencia.';
        return null;
      }
      final List<double> features = ImageUtils.extractFacialFeatureVector(
        decoded,
        scanResolution: AppConstants.highResFacialScanSize,
      );
      final Uint8List previewBytes = ImageUtils.encodeProcessedPreviewPng(
        decoded,
        targetSize: _processedImageSize,
      );
      return ConfiguredMoodProfile(
        emotion: emotion,
        featureVector: features,
        updatedAt: DateTime.now(),
        previewImageBytes: previewBytes,
        source: source,
      );
    } catch (e) {
      _errorMessage = 'Error al procesar imagen: $e';
      return null;
    } finally {
      try {
        if (await rawFile.exists()) {
          await rawFile.delete();
        }
      } catch (_) {}
    }
  }

  Future<void> _captureMoodReferenceWithActiveCamera(String emotion) async {
    if (_busyEmotion != null) {
      return;
    }
    setState(() {
      _busyEmotion = emotion;
      _errorMessage = null;
    });

    try {
      if (!widget.cameraService.isInitialized) {
        final bool ok = await widget.cameraService.initialize(
          preferredCameraName: _selectedCameraId,
        );
        if (!ok) {
          if (mounted) {
            setState(() {
              _errorMessage = widget.cameraService.lastError ??
                  'No se pudo inicializar la cámara activa.';
            });
          }
          return;
        }
      }

      String? capturedPath;
      final CameraController? liveController = widget.cameraService.controller;
      if (liveController != null &&
          liveController.value.isInitialized &&
          mounted) {
        capturedPath = await showDialog<String>(
          context: context,
          builder: (BuildContext dialogContext) {
            bool isCapturing = false;
            return StatefulBuilder(
              builder: (BuildContext context, StateSetter setDialogState) {
                final String emotionEs = AppConstants.translateEmotion(emotion);
                return AlertDialog(
                  title: Text('Capturar referencia: $emotionEs'),
                  content: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Text(
                        'Adopta la expresión de «$emotionEs» frente a la cámara activa y pulsa Capturar.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      const SizedBox(height: 12),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: SizedBox(
                          width: 240,
                          height: 300,
                          child: FittedBox(
                            fit: BoxFit.cover,
                            child: SizedBox(
                              width: liveController.value.previewSize?.height ??
                                  240,
                              height: liveController.value.previewSize?.width ??
                                  300,
                              child: CameraPreview(liveController),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  actions: <Widget>[
                    TextButton(
                      onPressed: isCapturing
                          ? null
                          : () => Navigator.of(dialogContext).pop(),
                      child: const Text('Cancelar'),
                    ),
                    FilledButton.icon(
                      onPressed: isCapturing
                          ? null
                          : () async {
                              setDialogState(() {
                                isCapturing = true;
                              });
                              final String? path = await widget.cameraService
                                  .takePeriodicSnapshot();
                              if (dialogContext.mounted) {
                                Navigator.of(dialogContext).pop(path);
                              }
                            },
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Capturar'),
                    ),
                  ],
                );
              },
            );
          },
        );
      } else {
        capturedPath = await widget.cameraService.takePeriodicSnapshot();
      }

      if (capturedPath == null) {
        return;
      }

      final ConfiguredMoodProfile? profile = await _buildProfileFromImageFile(
        capturedPath,
        emotion: emotion,
        source: 'camera',
      );
      if (profile == null) {
        if (mounted) {
          setState(() {
            _errorMessage = widget.mlService?.lastError ??
                _errorMessage ??
                'No se pudo extraer la referencia facial de la captura.';
          });
        }
        return;
      }

      await widget.storageService.setConfiguredMoodProfile(profile);
      final Map<String, ConfiguredMoodProfile> updated =
          await widget.storageService.getConfiguredMoodProfiles();
      if (!mounted) {
        return;
      }
      setState(() {
        _configuredMoods = updated;
      });
    } finally {
      if (mounted) {
        setState(() {
          _busyEmotion = null;
        });
      }
    }
  }

  Future<void> _pickImageFromDeviceStorage(String emotion) async {
    if (_busyEmotion != null) {
      return;
    }
    setState(() {
      _busyEmotion = emotion;
      _errorMessage = null;
    });

    try {
      final String? pickedPath =
          await _imagePickerChannel.invokeMethod<String>('pickExistingImage');
      if (pickedPath == null || pickedPath.isEmpty) {
        return;
      }

      final ConfiguredMoodProfile? profile = await _buildProfileFromImageFile(
        pickedPath,
        emotion: emotion,
        source: 'existing',
      );
      if (profile == null) {
        if (mounted) {
          setState(() {
            _errorMessage = widget.mlService?.lastError ??
                _errorMessage ??
                'No se pudo procesar la foto existente seleccionada.';
          });
        }
        return;
      }

      await widget.storageService.setConfiguredMoodProfile(profile);
      final Map<String, ConfiguredMoodProfile> updated =
          await widget.storageService.getConfiguredMoodProfiles();
      if (!mounted) {
        return;
      }
      setState(() {
        _configuredMoods = updated;
      });
    } on PlatformException catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Error al abrir el selector de fotos: ${e.message}';
        });
      }
    } on MissingPluginException {
      if (mounted) {
        setState(() {
          _errorMessage = 'Selector de archivos no disponible en este entorno.';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _busyEmotion = null;
        });
      }
    }
  }

  Future<void> _applyExistingSampleAsReference(
    String emotion,
    TrainingSample sample,
  ) async {
    List<double> features = sample.featureVector;
    if (features.isEmpty && sample.processedImageBytes != null) {
      final img.Image? decoded =
          ImageUtils.decodeBytes(sample.processedImageBytes!);
      if (decoded != null) {
        features = ImageUtils.extractFacialFeatureVector(
          decoded,
          scanResolution: AppConstants.highResFacialScanSize,
        );
      }
    }
    if (features.isEmpty) {
      setState(() {
        _errorMessage = 'La imagen seleccionada no contiene vector facial.';
      });
      return;
    }

    final ConfiguredMoodProfile profile = ConfiguredMoodProfile(
      emotion: emotion,
      featureVector: features,
      updatedAt: DateTime.now(),
      previewImageBytes: sample.processedImageBytes,
      source: 'existing',
    );
    await widget.storageService.setConfiguredMoodProfile(profile);
    final Map<String, ConfiguredMoodProfile> updated =
        await widget.storageService.getConfiguredMoodProfiles();
    if (!mounted) {
      return;
    }
    setState(() {
      _configuredMoods = updated;
      _errorMessage = null;
    });
  }

  Future<void> _selectExistingPhotoForMood(String emotion) async {
    if (widget.existingSamples.isEmpty) {
      await _pickImageFromDeviceStorage(emotion);
      return;
    }

    final String emotionEs = AppConstants.translateEmotion(emotion);
    await showModalBottomSheet<void>(
      context: context,
      builder: (BuildContext sheetContext) {
        final ThemeData theme = Theme.of(sheetContext);
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: <Widget>[
                Text(
                  'Seleccionar foto existente para $emotionEs',
                  style: theme.textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                ListTile(
                  leading: const Icon(Icons.folder_open),
                  title: const Text('Elegir foto de la galería / dispositivo'),
                  subtitle: const Text(
                    'Abrir el selector del sistema para escoger una foto almacenada',
                  ),
                  onTap: () {
                    Navigator.of(sheetContext).pop();
                    _pickImageFromDeviceStorage(emotion);
                  },
                ),
                const Divider(),
                Text(
                  'O elegir una captura existente de la sesión (${widget.existingSamples.length}):',
                  style: theme.textTheme.labelMedium,
                ),
                const SizedBox(height: 8),
                SizedBox(
                  height: 86,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: widget.existingSamples.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 10),
                    itemBuilder: (BuildContext context, int index) {
                      final TrainingSample item = widget.existingSamples[index];
                      return InkWell(
                        onTap: () {
                          Navigator.of(sheetContext).pop();
                          _applyExistingSampleAsReference(emotion, item);
                        },
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: <Widget>[
                            Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                border: Border.all(
                                  color: theme.colorScheme.outlineVariant,
                                ),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(7),
                                child: item.processedImageBytes != null
                                    ? Image.memory(
                                        item.processedImageBytes!,
                                        fit: BoxFit.cover,
                                        filterQuality: FilterQuality.none,
                                      )
                                    : const Icon(Icons.face),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              AppConstants.translateEmotion(
                                item.predictedEmotion,
                              ),
                              style: theme.textTheme.bodySmall,
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _removeMoodReference(String emotion) async {
    await widget.storageService.removeConfiguredMoodProfile(emotion);
    final Map<String, ConfiguredMoodProfile> updated =
        await widget.storageService.getConfiguredMoodProfiles();
    if (!mounted) {
      return;
    }
    setState(() {
      _configuredMoods = updated;
    });
  }

  Widget _buildMoodDefinitionsCard(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final int definedCount = _configuredMoods.length;

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Icon(
                  Icons.face_retouching_natural,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Definición de estados de ánimo de referencia',
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                Chip(
                  visualDensity: VisualDensity.compact,
                  label: Text(
                    '$definedCount/${AppConstants.emotionLabels.length} definidos',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Asigna una foto existente o tómala con la cámara activa para cada estado de ánimo. Estas imágenes se usarán como referencia directa para establecer el estado de ánimo resultante.',
              style: theme.textTheme.bodySmall,
            ),
            const Divider(height: 24),
            ...AppConstants.emotionLabels.map((String emotion) {
              final ConfiguredMoodProfile? profile = _configuredMoods[emotion];
              final bool isBusy = _busyEmotion == emotion;
              final String emotionEs = AppConstants.translateEmotion(emotion);
              final String statusText = profile == null
                  ? 'Sin imagen de referencia'
                  : (profile.source == 'camera'
                      ? 'Referencia activa (Cámara activa)'
                      : 'Referencia activa (Foto existente)');

              return Padding(
                padding: const EdgeInsets.only(bottom: 12.0),
                child: Container(
                  padding: const EdgeInsets.all(10.0),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: profile != null
                          ? theme.colorScheme.primary.withValues(alpha: 0.5)
                          : theme.colorScheme.outlineVariant,
                    ),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Row(
                        children: <Widget>[
                          Container(
                            width: 56,
                            height: 56,
                            decoration: BoxDecoration(
                              color: Colors.black26,
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: theme.colorScheme.outlineVariant,
                              ),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(7),
                              child: profile?.previewImageBytes != null
                                  ? Image.memory(
                                      profile!.previewImageBytes!,
                                      fit: BoxFit.cover,
                                      filterQuality: FilterQuality.none,
                                    )
                                  : Icon(
                                      Icons.person_outline,
                                      color: theme.colorScheme.outline,
                                    ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: <Widget>[
                                Text(
                                  emotionEs,
                                  style: theme.textTheme.titleSmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  statusText,
                                  style: theme.textTheme.bodySmall?.copyWith(
                                    color: profile != null
                                        ? Colors.green
                                        : theme.colorScheme.secondary,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          if (isBusy)
                            const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          else if (profile != null)
                            IconButton(
                              icon: const Icon(Icons.delete_outline),
                              tooltip: 'Borrar referencia de $emotionEs',
                              onPressed: () => _removeMoodReference(emotion),
                            ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: OutlinedButton.icon(
                              key: ValueKey<String>(
                                  'mood_existing_btn_$emotion'),
                              onPressed: isBusy
                                  ? null
                                  : () => _selectExistingPhotoForMood(emotion),
                              icon: const Icon(
                                Icons.photo_library_outlined,
                                size: 18,
                              ),
                              label: const Text('Foto existente'),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: FilledButton.tonalIcon(
                              key: ValueKey<String>('mood_camera_btn_$emotion'),
                              onPressed: isBusy
                                  ? null
                                  : () => _captureMoodReferenceWithActiveCamera(
                                        emotion,
                                      ),
                              icon: const Icon(
                                Icons.camera_alt_outlined,
                                size: 18,
                              ),
                              label: const Text('Cámara activa'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  String _formatLensDirection(CameraLensDirection direction) {
    switch (direction) {
      case CameraLensDirection.front:
        return 'Cámara frontal';
      case CameraLensDirection.back:
        return 'Cámara trasera';
      case CameraLensDirection.external:
        return 'Cámara externa';
    }
  }

  @override
  Widget build(BuildContext context) {
    final int currentMinutes = (_intervalSeconds / 60).round().clamp(1, 60);
    final int selectedSecondsOption =
        _secondsOptions.contains(_intervalSeconds) ? _intervalSeconds : 30;
    final int selectedMinutesOption =
        _minutesOptions.contains(currentMinutes) ? currentMinutes : 1;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuración'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: <Widget>[
          Card(
            child: SwitchListTile(
              secondary: const Icon(Icons.model_training),
              title: const Text('Modo Entrenamiento'),
              subtitle: const Text(
                'Permite inspeccionar cada imagen tratada y confirmar o corregir el estado de ánimo con un desplegable.',
              ),
              value: _trainingModeEnabled,
              onChanged: _toggleTrainingMode,
            ),
          ),
          const SizedBox(height: 12),
          _buildMoodDefinitionsCard(context),
          const SizedBox(height: 12),
          Card(
            child: ListTile(
              leading: const Icon(Icons.photo_size_select_large),
              title: const Text('Resolución de imagen tratada'),
              subtitle: const Text(
                'Tamaño del recorte facial procesado. El modelo actual sigue recibiendo su entrada nativa de 48x48.',
              ),
              trailing: DropdownButton<int>(
                value: _processedImageSize,
                items: AppConstants.processedImageSizes
                    .map(
                      (int px) => DropdownMenuItem<int>(
                        value: px,
                        child: Text('${px}x$px'),
                      ),
                    )
                    .toList(),
                onChanged: _updateProcessedImageSize,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  SwitchListTile(
                    title: const Text('Configurar intervalo en minutos'),
                    subtitle: Text(
                      _useMinutesUnit
                          ? 'Modo: Minutos (ideal para ahorrar batería y reducir temperatura)'
                          : 'Modo: Segundos (mínimo ${AppConstants.minCaptureIntervalSeconds} s para Snapdragon 660)',
                    ),
                    value: _useMinutesUnit,
                    onChanged: _toggleUnit,
                  ),
                  const Divider(),
                  ListTile(
                    title: const Text('Intervalo de captura'),
                    subtitle:
                        Text('Periodo activo: $_intervalSeconds segundos'),
                    trailing: _useMinutesUnit
                        ? DropdownButton<int>(
                            value: selectedMinutesOption,
                            items: _minutesOptions
                                .map(
                                  (int min) => DropdownMenuItem<int>(
                                    value: min,
                                    child: Text('$min min'),
                                  ),
                                )
                                .toList(),
                            onChanged: (int? min) {
                              if (min != null) {
                                _updateIntervalSeconds(min * 60);
                              }
                            },
                          )
                        : DropdownButton<int>(
                            value: selectedSecondsOption,
                            items: _secondsOptions
                                .map(
                                  (int sec) => DropdownMenuItem<int>(
                                    value: sec,
                                    child: Text('$sec s'),
                                  ),
                                )
                                .toList(),
                            onChanged: (int? sec) {
                              if (sec != null) {
                                _updateIntervalSeconds(sec);
                              }
                            },
                          ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: ListTile(
              title: const Text('Sensor de cámara'),
              subtitle: const Text(
                'Se recomienda la cámara frontal para la captura periódica de expresiones faciales.',
              ),
              trailing: _cameras.isEmpty
                  ? const Text('Predeterminada (Frontal)')
                  : DropdownButton<String>(
                      value: _cameras.any(
                        (CameraDescription c) => c.name == _selectedCameraId,
                      )
                          ? _selectedCameraId
                          : _cameras.first.name,
                      items: _cameras
                          .map(
                            (CameraDescription cam) => DropdownMenuItem<String>(
                              value: cam.name,
                              child: Text(
                                '${_formatLensDirection(cam.lensDirection)} (#${cam.name})',
                              ),
                            ),
                          )
                          .toList(),
                      onChanged: _updateCamera,
                    ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: ListTile(
              title: const Text('Normalización del tensor'),
              subtitle: const Text(
                'Escalado de píxeles en coma flotante aplicado al recorte facial.',
              ),
              trailing: DropdownButton<NormalizationRange>(
                value: _normalization,
                items: const <DropdownMenuItem<NormalizationRange>>[
                  DropdownMenuItem<NormalizationRange>(
                    value: NormalizationRange.zeroToOne,
                    child: Text('[0.0, 1.0]'),
                  ),
                  DropdownMenuItem<NormalizationRange>(
                    value: NormalizationRange.minusOneToOne,
                    child: Text('[-1.0, 1.0]'),
                  ),
                ],
                onChanged: _updateNormalization,
              ),
            ),
          ),
          if (_errorMessage != null) ...<Widget>[
            const SizedBox(height: 12),
            Text(
              _errorMessage!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ],
        ],
      ),
    );
  }
}
