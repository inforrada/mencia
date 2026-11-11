import 'dart:async';
import 'package:flutter/material.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import '../../core/constants/app_constants.dart';
import '../../core/utils/image_utils.dart';
import '../../services/camera_service.dart';
import '../../services/ml_service.dart';
import '../../services/storage_service.dart';
import '../widgets/mood_status_card.dart';
import '../widgets/training_review_card.dart';
import 'history_screen.dart';
import 'settings_screen.dart';

/// Main dashboard displaying live tracker status, Training Mode validation
/// ("Modo Entrenamiento") with treated image preview & confirmation/correction,
/// last detected mood (plus secondary moods >= 5%), active validated reference
/// prototypes, total snapshot counter, and navigation.
class HomeScreen extends StatefulWidget {
  const HomeScreen({
    super.key,
    required this.cameraService,
    required this.mlService,
    required this.storageService,
  });

  final CameraService cameraService;
  final MlService mlService;
  final StorageService storageService;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Timer? _periodicTimer;
  bool _isTracking = false;
  bool _isProcessing = false;
  bool _dimScreenOverlay = false;
  bool _isTrainingMode = false;
  int _intervalSeconds = AppConstants.defaultCaptureIntervalSeconds;
  NormalizationRange _normalization = NormalizationRange.zeroToOne;
  int _processedImageSize = AppConstants.defaultProcessedImageSize;
  int _inferenceCount = 0;
  String? _lastEmotion;
  double? _lastConfidence;
  Map<String, double> _lastProbabilities = const <String, double>{};
  int? _lastProcessingDurationMs;
  List<ValidatedMoodReference> _validatedReferences =
      <ValidatedMoodReference>[];
  String? _errorMessage;

  /// In-memory queue of processed images captured for review in Training Mode.
  final List<TrainingSample> _trainingSamples = <TrainingSample>[];

  @override
  void initState() {
    super.initState();
    _loadConfiguration();
  }

  Future<void> _loadConfiguration() async {
    final int savedInterval =
        await widget.storageService.getCaptureIntervalSeconds();
    final NormalizationRange savedNorm =
        await widget.storageService.getNormalizationRange();
    final bool savedTrainingMode =
        await widget.storageService.getTrainingModeEnabled();
    final int savedImageSize =
        await widget.storageService.getProcessedImageSize();
    final List<MoodRecord> history =
        await widget.storageService.getMoodHistory();
    final List<ValidatedMoodReference> references =
        await widget.storageService.getValidatedMoodReferences();
    if (!mounted) {
      return;
    }
    setState(() {
      _intervalSeconds = savedInterval;
      _normalization = savedNorm;
      _isTrainingMode = savedTrainingMode;
      _processedImageSize = savedImageSize;
      _inferenceCount = history.length;
      _validatedReferences = List<ValidatedMoodReference>.of(references);
      _recalculatePendingSamplesWithReferences();
      if (history.isNotEmpty) {
        _lastEmotion = history.first.emotion;
        _lastConfidence = history.first.confidence;
        _lastProbabilities = history.first.probabilities;
        _lastProcessingDurationMs = history.first.processingDurationMs;
      } else {
        _lastEmotion = null;
        _lastConfidence = null;
        _lastProbabilities = const <String, double>{};
        _lastProcessingDurationMs = null;
      }
    });
  }

  Future<void> _toggleTrainingMode(bool enabled) async {
    await widget.storageService.setTrainingModeEnabled(enabled);
    if (!mounted) {
      return;
    }
    setState(() {
      _isTrainingMode = enabled;
    });
  }

  Future<void> _toggleTracking() async {
    if (_isTracking) {
      await _stopTracking();
    } else {
      await _startTracking();
    }
  }

  Future<void> _startTracking() async {
    setState(() {
      _errorMessage = null;
    });

    final String? preferredCamera =
        await widget.storageService.getSelectedCameraId();
    final bool cameraReady = await widget.cameraService.initialize(
      preferredCameraName: preferredCamera,
    );
    if (!cameraReady) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = widget.cameraService.lastError;
      });
      return;
    }

    final bool mlReady = await widget.mlService.initialize();
    if (!mlReady) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = widget.mlService.lastError;
      });
      return;
    }

    try {
      await WakelockPlus.enable();
    } catch (e) {
      _errorMessage = 'Aviso de pantalla activa: $e';
    }

    _periodicTimer?.cancel();
    _periodicTimer = Timer.periodic(
      Duration(seconds: _intervalSeconds),
      (_) => _captureAndInfer(),
    );

    if (!mounted) {
      return;
    }
    setState(() {
      _isTracking = true;
    });

    // Trigger initial snapshot immediately upon starting
    await _captureAndInfer();
  }

  Future<void> _stopTracking() async {
    _periodicTimer?.cancel();
    _periodicTimer = null;
    try {
      await WakelockPlus.disable();
    } catch (_) {}
    if (!mounted) {
      return;
    }
    setState(() {
      _isTracking = false;
      _dimScreenOverlay = false;
    });
  }

  Future<void> _captureAndInfer() async {
    if (_isProcessing) {
      return;
    }
    setState(() {
      _isProcessing = true;
    });

    try {
      if (!widget.cameraService.isInitialized) {
        final String? preferredCamera =
            await widget.storageService.getSelectedCameraId();
        final bool ok = await widget.cameraService.initialize(
          preferredCameraName: preferredCamera,
        );
        if (!ok) {
          if (mounted) {
            setState(() {
              _errorMessage = widget.cameraService.lastError;
            });
          }
          return;
        }
      }
      if (!widget.mlService.isReady) {
        final bool ok = await widget.mlService.initialize();
        if (!ok) {
          if (mounted) {
            setState(() {
              _errorMessage = widget.mlService.lastError;
            });
          }
          return;
        }
      }

      final Stopwatch captureStopwatch = Stopwatch()..start();
      final String? path = await widget.cameraService.takePeriodicSnapshot();
      captureStopwatch.stop();
      if (path == null) {
        if (!mounted) {
          return;
        }
        setState(() {
          _errorMessage = widget.cameraService.lastError;
        });
        return;
      }

      final EmotionInferenceResult? result =
          await widget.mlService.processImageFile(
        path,
        normalization: _normalization,
        processedImageSize: _processedImageSize,
        scanResolution: AppConstants.highResFacialScanSize,
        captureDurationMs: captureStopwatch.elapsedMilliseconds,
        validatedReferences: _validatedReferences,
      );
      if (!mounted) {
        return;
      }

      if (result != null) {
        if (_isTrainingMode) {
          final TrainingSample sample = TrainingSample(
            id: result.timestamp.microsecondsSinceEpoch.toString(),
            predictedEmotion: result.label,
            confidence: result.confidence,
            timestamp: result.timestamp,
            probabilities: result.probabilities,
            processedImageBytes: result.processedImageBytes,
            imageSize: result.processedImageSize,
            modelInputSize: result.modelInputSize,
            scanResolution: result.scanResolution,
            processingDurationMs: result.processingDurationMs,
            featureVector: result.featureVector,
            rawTfliteScores: result.rawTfliteScores,
          );
          setState(() {
            _trainingSamples.insert(0, sample);
            if (_trainingSamples.length > 30) {
              _trainingSamples.removeLast();
            }
            _lastEmotion = result.label;
            _lastConfidence = result.confidence;
            _lastProbabilities = result.probabilities;
            _lastProcessingDurationMs = result.processingDurationMs;
            _errorMessage = null;
          });
        } else {
          await widget.storageService.appendMoodRecord(
            MoodRecord(
              emotion: result.label,
              confidence: result.confidence,
              timestamp: result.timestamp,
              predictedEmotion: result.label,
              probabilities: result.probabilities,
              featureVector: result.featureVector,
              processingDurationMs: result.processingDurationMs,
            ),
          );
          if (!mounted) {
            return;
          }
          setState(() {
            _inferenceCount++;
            _lastEmotion = result.label;
            _lastConfidence = result.confidence;
            _lastProbabilities = result.probabilities;
            _lastProcessingDurationMs = result.processingDurationMs;
            _errorMessage = null;
          });
        }
      } else {
        setState(() {
          _errorMessage = widget.mlService.lastError ??
              'No se ha detectado ningún rostro en la captura (foto temporal eliminada).';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  /// Re-evaluates pending training samples whenever a new validated mood is
  /// added to [_validatedReferences].
  void _recalculatePendingSamplesWithReferences() {
    for (final TrainingSample pending in _trainingSamples) {
      if (pending.status != TrainingSampleStatus.pending ||
          pending.featureVector.isEmpty) {
        continue;
      }
      final List<double> updatedList =
          ImageUtils.computePositionAndReferenceProbabilities(
        tfliteScores: pending.rawTfliteScores,
        featureVector: pending.featureVector,
        validatedReferences: _validatedReferences,
      );
      int maxIdx = 0;
      double maxScore = updatedList[0];
      final Map<String, double> updatedMap = <String, double>{};
      for (int i = 0; i < AppConstants.emotionLabels.length; i++) {
        final String label = AppConstants.emotionLabels[i];
        final double score = updatedList[i];
        updatedMap[label] = score;
        if (score > maxScore) {
          maxScore = score;
          maxIdx = i;
        }
      }
      pending.predictedEmotion = AppConstants.emotionLabels[maxIdx];
      pending.confidence = maxScore;
      pending.probabilities = updatedMap;
    }
  }

  Future<void> _handleConfirmSample(TrainingSample sample) async {
    final bool wasPending = sample.status == TrainingSampleStatus.pending;
    setState(() {
      sample.status = TrainingSampleStatus.confirmed;
      sample.confirmedEmotion = sample.predictedEmotion;
      if (sample.featureVector.isNotEmpty) {
        _validatedReferences = <ValidatedMoodReference>[
          ValidatedMoodReference(
            emotion: sample.predictedEmotion,
            featureVector: sample.featureVector,
          ),
          ..._validatedReferences,
        ];
        _recalculatePendingSamplesWithReferences();
      }
      _lastEmotion = sample.predictedEmotion;
      _lastConfidence = sample.confidence;
      _lastProbabilities = sample.probabilities;
      _lastProcessingDurationMs = sample.processingDurationMs;
    });

    await widget.storageService.appendMoodRecord(
      MoodRecord(
        emotion: sample.predictedEmotion,
        confidence: sample.confidence,
        timestamp: sample.timestamp,
        predictedEmotion: sample.predictedEmotion,
        isTrainingVerified: true,
        wasCorrected: false,
        probabilities: sample.probabilities,
        featureVector: sample.featureVector,
        processingDurationMs: sample.processingDurationMs,
      ),
    );
    if (!mounted) {
      return;
    }
    if (wasPending) {
      setState(() {
        _inferenceCount++;
      });
    }
  }

  Future<void> _handleCorrectSample(
    TrainingSample sample,
    String correctedEmotion,
  ) async {
    final bool wasPending = sample.status == TrainingSampleStatus.pending;
    final bool isDifferent = correctedEmotion != sample.predictedEmotion;
    setState(() {
      sample.status = isDifferent
          ? TrainingSampleStatus.corrected
          : TrainingSampleStatus.confirmed;
      sample.confirmedEmotion = correctedEmotion;
      if (sample.featureVector.isNotEmpty) {
        _validatedReferences = <ValidatedMoodReference>[
          ValidatedMoodReference(
            emotion: correctedEmotion,
            featureVector: sample.featureVector,
          ),
          ..._validatedReferences,
        ];
        _recalculatePendingSamplesWithReferences();
      }
      _lastEmotion = correctedEmotion;
      _lastConfidence = sample.confidence;
      _lastProbabilities = sample.probabilities;
      _lastProcessingDurationMs = sample.processingDurationMs;
    });

    await widget.storageService.appendMoodRecord(
      MoodRecord(
        emotion: correctedEmotion,
        confidence: sample.confidence,
        timestamp: sample.timestamp,
        predictedEmotion: sample.predictedEmotion,
        isTrainingVerified: true,
        wasCorrected: isDifferent,
        probabilities: sample.probabilities,
        featureVector: sample.featureVector,
        processingDurationMs: sample.processingDurationMs,
      ),
    );
    if (!mounted) {
      return;
    }
    if (wasPending) {
      setState(() {
        _inferenceCount++;
      });
    }
  }

  @override
  void dispose() {
    _periodicTimer?.cancel();
    WakelockPlus.disable();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: <Widget>[
        Scaffold(
          appBar: AppBar(
            title: const Text(AppConstants.appName),
            actions: <Widget>[
              if (_isTracking)
                IconButton(
                  icon: const Icon(Icons.brightness_low),
                  tooltip: 'Atenuar pantalla (ahorro de batería)',
                  onPressed: () {
                    setState(() {
                      _dimScreenOverlay = true;
                    });
                  },
                ),
              IconButton(
                icon: const Icon(Icons.history),
                tooltip: 'Historial de estados de ánimo',
                onPressed: () async {
                  await Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => HistoryScreen(
                        storageService: widget.storageService,
                      ),
                    ),
                  );
                  await _loadConfiguration();
                },
              ),
              IconButton(
                icon: const Icon(Icons.settings),
                tooltip: 'Configuración',
                onPressed: () async {
                  await Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => SettingsScreen(
                        storageService: widget.storageService,
                        cameraService: widget.cameraService,
                        mlService: widget.mlService,
                        existingSamples: _trainingSamples,
                      ),
                    ),
                  );
                  await _loadConfiguration();
                },
              ),
            ],
          ),
          body: SafeArea(
            child: Column(
              children: <Widget>[
                Expanded(
                  child: ListView(
                    padding: const EdgeInsets.all(16.0),
                    children: <Widget>[
                      Card(
                        child: SwitchListTile(
                          secondary: const Icon(Icons.model_training),
                          title: const Text('Modo Entrenamiento'),
                          subtitle: const Text(
                            'Visualizar imágenes tratadas y confirmar o corregir el estado de ánimo decidido',
                          ),
                          value: _isTrainingMode,
                          onChanged: _toggleTrainingMode,
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (_isTrainingMode) ...<Widget>[
                        TrainingReviewCard(
                          samples: _trainingSamples,
                          onConfirmSample: _handleConfirmSample,
                          onCorrectSample: _handleCorrectSample,
                        ),
                        const SizedBox(height: 12),
                      ],
                      MoodStatusCard(
                        isTracking: _isTracking,
                        lastEmotion: _lastEmotion,
                        lastConfidence: _lastConfidence,
                        lastProbabilities: _lastProbabilities,
                        lastProcessingDurationMs: _lastProcessingDurationMs,
                        scanResolution: AppConstants.highResFacialScanSize,
                        inferenceCount: _inferenceCount,
                        validatedReferenceCount: _validatedReferences.length,
                        errorMessage: _errorMessage,
                      ),
                      const SizedBox(height: 12),
                      Card(
                        child: ListTile(
                          leading: const Icon(Icons.memory),
                          title: const Text(
                            'Protección térmica Snapdragon 660',
                          ),
                          subtitle: Text(
                            'Intervalo: $_intervalSeconds s • Hilos CPU TFLite: ${AppConstants.tfliteCpuThreads} • 100 % sin conexión',
                          ),
                          trailing: _isProcessing
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : null,
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16.0, 8.0, 16.0, 16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: <Widget>[
                      OutlinedButton.icon(
                        onPressed: _isProcessing ? null : _captureAndInfer,
                        icon: const Icon(Icons.camera_alt_outlined),
                        label: const Text('Capturar ahora'),
                      ),
                      const SizedBox(height: 10),
                      FilledButton.icon(
                        onPressed: _toggleTracking,
                        icon: Icon(_isTracking ? Icons.stop : Icons.play_arrow),
                        label: Text(
                          _isTracking
                              ? 'Detener seguimiento'
                              : 'Iniciar seguimiento',
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        if (_dimScreenOverlay)
          Positioned.fill(
            child: GestureDetector(
              onTap: () {
                setState(() {
                  _dimScreenOverlay = false;
                });
              },
              child: ColoredBox(
                color: Colors.black.withValues(alpha: 0.94),
                child: const Center(
                  child: Text(
                    'Pantalla activa (Modo de bajo brillo)\nToca en cualquier lugar para restaurar el brillo',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white24,
                      fontSize: 14,
                      decoration: TextDecoration.none,
                    ),
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}
