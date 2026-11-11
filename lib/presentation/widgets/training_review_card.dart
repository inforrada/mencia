import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../core/constants/app_constants.dart';
import '../../core/utils/image_utils.dart';

/// Validation state of a processed facial snapshot in Training Mode.
enum TrainingSampleStatus {
  pending,
  confirmed,
  corrected,
}

/// Represents an in-memory processed face sample awaiting or having completed
/// user confirmation/correction in Training Mode.
class TrainingSample {
  TrainingSample({
    required this.id,
    required this.predictedEmotion,
    required this.confidence,
    required this.timestamp,
    this.probabilities = const <String, double>{},
    this.processedImageBytes,
    this.imageSize = AppConstants.defaultProcessedImageSize,
    this.modelInputSize = AppConstants.defaultInputSize,
    this.scanResolution = AppConstants.highResFacialScanSize,
    this.processingDurationMs,
    this.featureVector = const <double>[],
    this.rawTfliteScores = const <double>[],
    this.status = TrainingSampleStatus.pending,
    this.confirmedEmotion,
  });

  final String id;
  String predictedEmotion;
  double confidence;
  final DateTime timestamp;
  Map<String, double> probabilities;
  final Uint8List? processedImageBytes;

  /// Side length in pixels of [processedImageBytes].
  final int imageSize;

  /// Side length in pixels of the tensor the model actually received.
  final int modelInputSize;

  /// Side length in pixels of the high-resolution facial position scan grid.
  final int scanResolution;

  /// Total time in milliseconds spent capturing and processing this sample.
  final int? processingDurationMs;

  /// Extracted 14-D facial position & spatial appearance feature vector.
  final List<double> featureVector;

  /// Raw 7-class scores from the TFLite interpreter.
  final List<double> rawTfliteScores;

  TrainingSampleStatus status;
  String? confirmedEmotion;

  /// Secondary moods with at least [minProbability] (default 5%), excluding [predictedEmotion].
  List<MapEntry<String, double>> secondaryMoods({
    double minProbability = AppConstants.minSecondaryMoodProbability,
  }) {
    return ImageUtils.filterSignificantMoods(
      probabilities,
      minProbability: minProbability,
      excludeEmotion: predictedEmotion,
    );
  }
}

/// Interactive Training Mode widget that displays the preprocessed face images
/// ("imágenes tratadas"), the model's decided Mood ("Mood decidido"), all
/// secondary candidate moods with >= 5% probability, and lets the user Confirm
/// or Reject (selecting the true Mood from a dropdown).
class TrainingReviewCard extends StatefulWidget {
  const TrainingReviewCard({
    super.key,
    required this.samples,
    required this.onConfirmSample,
    required this.onCorrectSample,
  });

  final List<TrainingSample> samples;
  final Future<void> Function(TrainingSample sample) onConfirmSample;
  final Future<void> Function(TrainingSample sample, String correctedEmotion)
      onCorrectSample;

  @override
  State<TrainingReviewCard> createState() => _TrainingReviewCardState();
}

class _TrainingReviewCardState extends State<TrainingReviewCard> {
  int _selectedIndex = 0;
  bool _isRejecting = false;
  String _selectedCorrection = AppConstants.emotionLabels.first;

  @override
  void didUpdateWidget(covariant TrainingReviewCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.samples.isEmpty) {
      _selectedIndex = 0;
      _isRejecting = false;
    } else if (_selectedIndex >= widget.samples.length) {
      _selectedIndex = 0;
      _isRejecting = false;
    } else if (widget.samples.length > oldWidget.samples.length) {
      // Jump to newest captured sample
      _selectedIndex = 0;
      _isRejecting = false;
      _initDefaultCorrection(widget.samples.first);
    }
  }

  void _initDefaultCorrection(TrainingSample sample) {
    final List<MapEntry<String, double>> secondaries = sample.secondaryMoods();
    if (secondaries.isNotEmpty) {
      _selectedCorrection = secondaries.first.key;
      return;
    }
    final String fallback = AppConstants.emotionLabels.firstWhere(
      (String label) => label != sample.predictedEmotion,
      orElse: () => AppConstants.emotionLabels.first,
    );
    _selectedCorrection = fallback;
  }

  void _selectSample(int index) {
    setState(() {
      _selectedIndex = index;
      _isRejecting = false;
      _initDefaultCorrection(widget.samples[index]);
    });
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final DateFormat timeFormat = DateFormat('HH:mm:ss');

    if (widget.samples.isEmpty) {
      return Card(
        elevation: 2,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: <Widget>[
              Icon(
                Icons.model_training,
                size: 36,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(height: 8),
              Text(
                'Modo Entrenamiento activo',
                style: theme.textTheme.titleMedium,
              ),
              const SizedBox(height: 4),
              Text(
                'Inicia el seguimiento o pulsa «Capturar ahora» para visualizar las imágenes tratadas, revisar el estado de ánimo decidido y confirmar o corregir la expresión.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall,
              ),
            ],
          ),
        ),
      );
    }

    final TrainingSample current = widget.samples[_selectedIndex];
    final List<MapEntry<String, double>> secondaryMoods =
        current.secondaryMoods();
    final int pendingCount = widget.samples
        .where((TrainingSample s) => s.status == TrainingSampleStatus.pending)
        .length;

    return Card(
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                Row(
                  children: <Widget>[
                    Icon(
                      Icons.model_training,
                      color: theme.colorScheme.primary,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Modo Entrenamiento',
                      style: theme.textTheme.titleMedium,
                    ),
                  ],
                ),
                Chip(
                  visualDensity: VisualDensity.compact,
                  label: Text(
                    pendingCount > 0
                        ? '$pendingCount pendientes'
                        : 'Todas validadas',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            // Navigation header across processed images
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                IconButton(
                  icon: const Icon(Icons.chevron_left),
                  tooltip: 'Imagen anterior',
                  onPressed: _selectedIndex > 0
                      ? () => _selectSample(_selectedIndex - 1)
                      : null,
                ),
                Text(
                  'Imagen tratada ${_selectedIndex + 1} de ${widget.samples.length} (${timeFormat.format(current.timestamp)})',
                  style: theme.textTheme.bodySmall,
                ),
                IconButton(
                  icon: const Icon(Icons.chevron_right),
                  tooltip: 'Siguiente imagen',
                  onPressed: _selectedIndex < widget.samples.length - 1
                      ? () => _selectSample(_selectedIndex + 1)
                      : null,
                ),
              ],
            ),
            if (widget.samples.length > 1) ...<Widget>[
              SizedBox(
                height: 54,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: widget.samples.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (BuildContext context, int index) {
                    final TrainingSample item = widget.samples[index];
                    final bool isSelected = index == _selectedIndex;
                    Color borderColor = Colors.transparent;
                    if (isSelected) {
                      borderColor = theme.colorScheme.primary;
                    } else if (item.status == TrainingSampleStatus.confirmed) {
                      borderColor = Colors.green;
                    } else if (item.status == TrainingSampleStatus.corrected) {
                      borderColor = Colors.orange;
                    }
                    return GestureDetector(
                      onTap: () => _selectSample(index),
                      child: Container(
                        width: 50,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: borderColor,
                            width: isSelected ? 2.5 : 1.5,
                          ),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: item.processedImageBytes != null
                              ? Image.memory(
                                  item.processedImageBytes!,
                                  fit: BoxFit.cover,
                                  filterQuality: FilterQuality.none,
                                )
                              : const Icon(Icons.face, size: 24),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 12),
            ],
            // Main Treated Image & Decided Mood Display
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: <Widget>[
                Container(
                  width: 112,
                  height: 112,
                  decoration: BoxDecoration(
                    color: Colors.black26,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: theme.colorScheme.outlineVariant),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(7),
                    child: current.processedImageBytes != null
                        ? Image.memory(
                            current.processedImageBytes!,
                            fit: BoxFit.contain,
                            filterQuality: FilterQuality.none,
                          )
                        : const Center(
                            child: Icon(Icons.image_not_supported_outlined),
                          ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Text(
                        'Estado de ánimo decidido:',
                        style: theme.textTheme.labelMedium,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        AppConstants.translateEmotion(current.predictedEmotion),
                        style: theme.textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.primary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Confianza: ${(current.confidence * 100).toStringAsFixed(1)}%',
                        style: theme.textTheme.bodyMedium,
                      ),
                      if (current.processingDurationMs != null) ...<Widget>[
                        const SizedBox(height: 4),
                        Row(
                          children: <Widget>[
                            Icon(
                              Icons.timer_outlined,
                              size: 15,
                              color: theme.colorScheme.primary,
                            ),
                            const SizedBox(width: 4),
                            Flexible(
                              child: Text(
                                'Tiempo de procesado: ${ImageUtils.formatProcessingDuration(current.processingDurationMs!)}',
                                style: theme.textTheme.bodySmall?.copyWith(
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                      const SizedBox(height: 4),
                      Text(
                        'Recorte facial +10% • ${current.imageSize}x${current.imageSize} • Escaneo: ${current.scanResolution}x${current.scanResolution}',
                        style: theme.textTheme.bodySmall,
                      ),
                      if (current.imageSize != current.modelInputSize)
                        Text(
                          'Modelo: ${current.modelInputSize}x${current.modelInputSize}',
                          style: theme.textTheme.bodySmall,
                        ),
                      const SizedBox(height: 6),
                      _buildStatusBadge(theme, current),
                    ],
                  ),
                ),
              ],
            ),
            if (secondaryMoods.isNotEmpty) ...<Widget>[
              const SizedBox(height: 12),
              Text(
                'Otros estados de ánimo posibles (≥5%):',
                style: theme.textTheme.labelMedium?.copyWith(
                  color: theme.colorScheme.secondary,
                ),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: secondaryMoods.map(
                  (MapEntry<String, double> entry) {
                    final String pct = (entry.value * 100).toStringAsFixed(1);
                    final String labelEs =
                        AppConstants.translateEmotion(entry.key);
                    return ActionChip(
                      visualDensity: VisualDensity.compact,
                      label: Text('$labelEs: $pct%'),
                      onPressed: () {
                        setState(() {
                          _isRejecting = true;
                          _selectedCorrection = entry.key;
                        });
                      },
                    );
                  },
                ).toList(),
              ),
            ],
            const SizedBox(height: 14),
            // Confirm / Reject Action Buttons
            Row(
              children: <Widget>[
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () async {
                      setState(() {
                        _isRejecting = false;
                      });
                      await widget.onConfirmSample(current);
                    },
                    icon: const Icon(Icons.check),
                    label: const Text('Confirmar'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      setState(() {
                        _isRejecting = !_isRejecting;
                        _initDefaultCorrection(current);
                      });
                    },
                    icon: const Icon(Icons.close),
                    label: const Text('Rechazar'),
                  ),
                ),
              ],
            ),
            // Dropdown selector when Rejected
            if (_isRejecting) ...<Widget>[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: <Widget>[
                    Text(
                      'Indica cuál es el estado de ánimo correcto:',
                      style: theme.textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<String>(
                      initialValue: _selectedCorrection,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                      ),
                      items: AppConstants.emotionLabels
                          .map(
                            (String emotion) => DropdownMenuItem<String>(
                              value: emotion,
                              child: Text(
                                AppConstants.translateEmotion(emotion),
                              ),
                            ),
                          )
                          .toList(),
                      onChanged: (String? value) {
                        if (value != null) {
                          setState(() {
                            _selectedCorrection = value;
                          });
                        }
                      },
                    ),
                    const SizedBox(height: 10),
                    FilledButton.tonalIcon(
                      onPressed: () async {
                        final String chosen = _selectedCorrection;
                        setState(() {
                          _isRejecting = false;
                        });
                        await widget.onCorrectSample(current, chosen);
                      },
                      icon: const Icon(Icons.save_outlined),
                      label: const Text('Guardar estado de ánimo correcto'),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBadge(ThemeData theme, TrainingSample sample) {
    switch (sample.status) {
      case TrainingSampleStatus.pending:
        return Text(
          'Estado: Pendiente de confirmar',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.secondary,
          ),
        );
      case TrainingSampleStatus.confirmed:
        return Row(
          children: <Widget>[
            const Icon(Icons.verified, size: 16, color: Colors.green),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                'Confirmado (${AppConstants.translateEmotion(sample.confirmedEmotion ?? sample.predictedEmotion)})',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: Colors.green,
                ),
              ),
            ),
          ],
        );
      case TrainingSampleStatus.corrected:
        return Row(
          children: <Widget>[
            const Icon(Icons.edit_note, size: 16, color: Colors.orange),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                'Corregido: ${AppConstants.translateEmotion(sample.confirmedEmotion)}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: Colors.orange,
                ),
              ),
            ),
          ],
        );
    }
  }
}
