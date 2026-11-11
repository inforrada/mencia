import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../core/constants/app_constants.dart';
import '../../core/utils/image_utils.dart';
import '../../services/storage_service.dart';
import '../widgets/mood_chart_widget.dart';

/// Displays the historical mood distribution chart and chronological timeline
/// of locally inferred mood records, including Training Mode confirmations,
/// corrections, and secondary moods with at least 5% probability.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({
    super.key,
    required this.storageService,
  });

  final StorageService storageService;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<MoodRecord> _records = const <MoodRecord>[];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final List<MoodRecord> records =
        await widget.storageService.getMoodHistory();
    if (!mounted) {
      return;
    }
    setState(() {
      _records = records;
      _isLoading = false;
    });
  }

  String _buildRecordSubtitle(MoodRecord item, DateFormat formatter) {
    final StringBuffer buffer = StringBuffer(formatter.format(item.timestamp));
    if (item.processingDurationMs != null) {
      buffer.write(
        ' • Procesado: ${ImageUtils.formatProcessingDuration(item.processingDurationMs!)}',
      );
    }
    if (item.isTrainingVerified) {
      if (item.wasCorrected && item.predictedEmotion != null) {
        buffer.write(
          ' • Corregido (Modelo: ${AppConstants.translateEmotion(item.predictedEmotion)} → Real: ${AppConstants.translateEmotion(item.emotion)})',
        );
      } else {
        buffer.write(' • Confirmado en entrenamiento');
      }
    }
    final List<MapEntry<String, double>> secondaries = item.secondaryMoods();
    if (secondaries.isNotEmpty) {
      final String formattedSecondaries = secondaries
          .map(
            (MapEntry<String, double> e) =>
                '${AppConstants.translateEmotion(e.key)} ${(e.value * 100).toStringAsFixed(1)}%',
          )
          .join(', ');
      buffer.write('\nOtros (≥5%): $formattedSecondaries');
    }
    return buffer.toString();
  }

  @override
  Widget build(BuildContext context) {
    final DateFormat formatter = DateFormat('yyyy-MM-dd HH:mm:ss');
    return Scaffold(
      appBar: AppBar(
        title: const Text('Historial de estados de ánimo'),
        actions: <Widget>[
          if (_records.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Borrar historial',
              onPressed: () async {
                await widget.storageService.clearMoodHistory();
                await _loadHistory();
              },
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _records.isEmpty
              ? const Center(
                  child: Text(
                    'Todavía no hay registros de estado de ánimo guardados.',
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16.0),
                  itemCount: _records.length + 1,
                  itemBuilder: (BuildContext context, int index) {
                    if (index == 0) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 16.0),
                        child: MoodChartWidget(records: _records),
                      );
                    }
                    final MoodRecord item = _records[index - 1];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 8.0),
                      child: ListTile(
                        isThreeLine: item.secondaryMoods().isNotEmpty,
                        leading: Icon(
                          item.isTrainingVerified
                              ? (item.wasCorrected
                                  ? Icons.edit_note
                                  : Icons.verified)
                              : Icons.face_retouching_natural,
                          color: item.isTrainingVerified
                              ? (item.wasCorrected
                                  ? Colors.orange
                                  : Colors.green)
                              : null,
                        ),
                        title:
                            Text(AppConstants.translateEmotion(item.emotion)),
                        subtitle: Text(_buildRecordSubtitle(item, formatter)),
                        trailing: Text(
                          '${(item.confidence * 100).toStringAsFixed(1)}%',
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}
