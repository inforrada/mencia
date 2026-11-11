import 'package:flutter/material.dart';
import '../../core/constants/app_constants.dart';
import '../../core/utils/image_utils.dart';

/// Displays the current tracking status, latest detected primary emotion,
/// processing duration, all secondary candidate emotions with at least 5%
/// probability, active validated reference count, and total inference count.
class MoodStatusCard extends StatelessWidget {
  const MoodStatusCard({
    super.key,
    required this.isTracking,
    required this.lastEmotion,
    required this.inferenceCount,
    this.lastConfidence,
    this.lastProbabilities = const <String, double>{},
    this.lastProcessingDurationMs,
    this.scanResolution = AppConstants.highResFacialScanSize,
    this.validatedReferenceCount = 0,
    this.errorMessage,
  });

  final bool isTracking;
  final String? lastEmotion;
  final double? lastConfidence;
  final Map<String, double> lastProbabilities;
  final int? lastProcessingDurationMs;
  final int scanResolution;
  final int inferenceCount;
  final int validatedReferenceCount;
  final String? errorMessage;

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final List<MapEntry<String, double>> secondaryMoods =
        ImageUtils.filterSignificantMoods(
      lastProbabilities,
      minProbability: AppConstants.minSecondaryMoodProbability,
      excludeEmotion: lastEmotion,
    );

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
                  isTracking ? Icons.bolt : Icons.pause_circle_outline,
                  color: isTracking ? Colors.green : Colors.grey,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    isTracking
                        ? 'Seguimiento activo (Sin conexión)'
                        : 'Seguimiento en pausa',
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                if (validatedReferenceCount > 0)
                  Chip(
                    visualDensity: VisualDensity.compact,
                    avatar: const Icon(Icons.psychology_alt, size: 16),
                    label: Text('$validatedReferenceCount ref.'),
                  ),
              ],
            ),
            const Divider(height: 24),
            Text(
              'Último estado de ánimo: ${AppConstants.translateEmotion(lastEmotion)}',
              style: theme.textTheme.headlineSmall,
            ),
            if (lastConfidence != null) ...<Widget>[
              const SizedBox(height: 4),
              Text(
                'Confianza: ${(lastConfidence! * 100).toStringAsFixed(1)}%',
                style: theme.textTheme.bodyMedium,
              ),
            ],
            if (lastProcessingDurationMs != null) ...<Widget>[
              const SizedBox(height: 4),
              Row(
                children: <Widget>[
                  Icon(
                    Icons.timer_outlined,
                    size: 16,
                    color: theme.colorScheme.primary,
                  ),
                  const SizedBox(width: 4),
                  Flexible(
                    child: Text(
                      'Tiempo de procesado: ${ImageUtils.formatProcessingDuration(lastProcessingDurationMs!)}',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 4),
            Text(
              'Escaneo de posición facial: ${scanResolution}x$scanResolution px (alta resolución)',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.secondary,
              ),
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
                    return Chip(
                      visualDensity: VisualDensity.compact,
                      label: Text('$labelEs: $pct%'),
                    );
                  },
                ).toList(),
              ),
            ],
            const SizedBox(height: 8),
            Text(
              'Capturas procesadas: $inferenceCount • Referencias activas: $validatedReferenceCount',
              style: theme.textTheme.bodyMedium,
            ),
            if (errorMessage != null) ...<Widget>[
              const SizedBox(height: 12),
              Text(
                errorMessage!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
