import 'package:flutter/material.dart';
import '../../core/constants/app_constants.dart';
import '../../services/storage_service.dart';

/// Displays an offline horizontal bar chart summarizing the frequency of each
/// detected emotion across stored [MoodRecord] entries.
class MoodChartWidget extends StatelessWidget {
  const MoodChartWidget({
    super.key,
    required this.records,
  });

  final List<MoodRecord> records;

  Map<String, int> _computeCounts() {
    final Map<String, int> counts = <String, int>{
      for (final String label in AppConstants.emotionLabels) label: 0,
    };
    for (final MoodRecord record in records) {
      counts[record.emotion] = (counts[record.emotion] ?? 0) + 1;
    }
    return counts;
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final Map<String, int> counts = _computeCounts();
    final int total = records.length;

    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Distribución de estados de ánimo ($total muestras)',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            ...AppConstants.emotionLabels.map((String label) {
              final int count = counts[label] ?? 0;
              final double ratio = total > 0 ? count / total : 0.0;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  children: <Widget>[
                    SizedBox(
                      width: 76,
                      child: Text(
                        AppConstants.translateEmotion(label),
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                    Expanded(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: ratio,
                          minHeight: 10,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 44,
                      child: Text(
                        '$count',
                        textAlign: TextAlign.end,
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }
}
