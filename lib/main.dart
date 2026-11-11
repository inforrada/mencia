import 'package:flutter/material.dart';
import 'core/constants/app_constants.dart';
import 'presentation/screens/home_screen.dart';
import 'services/camera_service.dart';
import 'services/ml_service.dart';
import 'services/storage_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MenciaApp());
}

/// Root widget for the offline Mencia facial emotion tracker.
class MenciaApp extends StatefulWidget {
  const MenciaApp({super.key});

  @override
  State<MenciaApp> createState() => _MenciaAppState();
}

class _MenciaAppState extends State<MenciaApp> {
  late final CameraService _cameraService = CameraService();
  late final MlService _mlService = MlService();
  late final StorageService _storageService = StorageService();

  @override
  void dispose() {
    _cameraService.dispose();
    _mlService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppConstants.appName,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: HomeScreen(
        cameraService: _cameraService,
        mlService: _mlService,
        storageService: _storageService,
      ),
    );
  }
}
