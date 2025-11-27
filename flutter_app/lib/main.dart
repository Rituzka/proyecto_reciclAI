import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle, SystemNavigator;
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';
import 'package:image_picker/image_picker.dart';

// -------------------- Config global simple --------------------
final _picker = ImagePicker();

class HistoryItem {
  final String path;     // ruta de la imagen guardada
  final String label;    // Amarillo / Verde / Azul / Gris
  final double conf;     // 0..1
  final DateTime ts;     // timestamp
  HistoryItem(this.path, this.label, this.conf, this.ts);
}

final List<HistoryItem> _history = [];

Future<String> _saveImageToGallery(XFile xf) async {
  // Carpeta pública "Pictures/ReciclAI"
  final dir = Directory('/storage/emulated/0/Pictures/ReciclAI');

  if (!await dir.exists()) {
    await dir.create(recursive: true);
  }

  final ts = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
  final dstPath = '${dir.path}/reciclai_$ts.jpg';

  await File(xf.path).copy(dstPath);
  return dstPath;
}


// -------------------- App --------------------
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(ReciclaiApp(cameras: cameras));
}

class ReciclaiApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const ReciclaiApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ReciclAI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: ReciclaiPage(cameras: cameras),
    );
  }
}

// -------------------- Pantalla principal --------------------
class ReciclaiPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const ReciclaiPage({super.key, required this.cameras});

  @override
  State<ReciclaiPage> createState() => _ReciclaiPageState();
}

class _ReciclaiPageState extends State<ReciclaiPage> {
  CameraController? _controller;
  Interpreter? _interpreter;
  List<String> _labels = const [];

  bool _isReady = false;
  bool _torchOn = false;
  String _result = '—';
  Color _barColor = Colors.black54;
  double _bannerOpacity = 1.0;

  static const _inputW = 224, _inputH = 224, _channels = 3;
  static const double _threshold = 0.60; // 60%

  final Map<String, Color> _containerColors = {
    'Amarillo': Colors.amber,
    'Verde'   : Colors.green,
    'Azul'    : Colors.blue,
    'Gris'    : Colors.grey,   // fallback
  };

  @override
  void initState() {
    super.initState();
    _initAll();
  }

  Future<void> _initAll() async {
    // Cámara (trasera si hay)
    final cam = widget.cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );
    _controller = CameraController(
      cam,
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.yuv420,
      enableAudio: false,
    );
    await _controller!.initialize();

    // Modelo (FLOAT32 con preprocess_input dentro → mandamos 0..255)
    final opts = InterpreterOptions()..threads = 2;
    try { opts.addDelegate(XNNPackDelegate()); } catch (_) {}
    _interpreter = await Interpreter.fromAsset(
      'assets/models/proyecto-reciclAI_quant.tflite',
      options: opts,
    );

    // Labels
    final txt = await rootBundle.loadString('assets/labels.txt');
    _labels = txt.split('\n').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();

    // Flash OFF de inicio
    try { await _controller!.setFlashMode(FlashMode.off); } catch (_) {}

    if (mounted) setState(() => _isReady = true);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  Future<void> _setFlash(FlashMode mode) async {
    try { await _controller?.setFlashMode(mode); } catch (_) {}
  }

  void _toggleTorch() async {
    _torchOn = !_torchOn;
    await _setFlash(_torchOn ? FlashMode.torch : FlashMode.off);
    if (mounted) setState(() {});
  }

  void _clearResult() {
    _result = '—';
    _barColor = Colors.black54;
    if (mounted) setState(() {});
  }

  // ---------- Clasificación: helper común para cámara/galería ----------
  Future<void> _classifyBytesAndUpdate(Uint8List bytes, {String? savedPath}) async {
    if (_interpreter == null) return;

    // decode + resize
    final img.Image? src = img.decodeImage(bytes);
    if (src == null) {
      setState(() => _result = 'Imagen inválida');
      return;
    }
    final img.Image resized = img.copyResize(src, width: _inputW, height: _inputH);

    // tensor FLOAT32 0..255
    final flat = Float32List(_inputW * _inputH * _channels);
    int k = 0;
    for (int y = 0; y < _inputH; y++) {
      for (int x = 0; x < _inputW; x++) {
        final px = resized.getPixel(x, y);
        flat[k++] = px.r.toDouble();
        flat[k++] = px.g.toDouble();
        flat[k++] = px.b.toDouble();
      }
    }
    final input = _reshape4D(flat, 1, _inputH, _inputW, _channels);

    // salida [1, numClasses]
    final outShape = _interpreter!.getOutputTensors().first.shape;
    final numClasses = outShape.isNotEmpty ? outShape.last : 3;
    final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

    // inferencia
    _interpreter!.run(input, output);

    final probs = output[0].cast<double>();
    // debugPrint('probs: $probs');

    int topI = 0; double topV = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > topV) { topV = probs[i]; topI = i; }
    }

    double conf = topV; // 0..1
    String label = (topI < _labels.length) ? _labels[topI] : 'Gris';
    if (conf < _threshold) label = 'Gris';

    // historial si corresponde
    if (savedPath != null) {
      _history.insert(0, HistoryItem(savedPath, label, conf, DateTime.now()));
    }

    final containerText = 'Contenedor ${label.toLowerCase()}';
    _barColor = _containerColors[label] ?? Colors.black54;

    setState(() {
      _result = '$containerText (${(conf * 100).toStringAsFixed(0)}%)';
      _bannerOpacity = 0.0;
    });
    await Future.delayed(const Duration(milliseconds: 50));
    if (mounted) setState(() => _bannerOpacity = 1.0);
  }

  // ---------- Cámara: capturar y clasificar ----------
  Future<void> _captureAndClassify() async {
    final ready = _controller?.value.isInitialized == true && _interpreter != null;
    if (!ready) return;

    try {
      final XFile xf = await _controller!.takePicture();
      final savedPath = await _saveImageToGallery(xf);


      // respeta el estado del flash tras capturar
      try {
        await _controller!.setFlashMode(_torchOn ? FlashMode.torch : FlashMode.off);
      } catch (_) {}

      final bytes = await File(xf.path).readAsBytes();
      await _classifyBytesAndUpdate(bytes, savedPath: savedPath);
    } catch (e) {
      setState(() => _result = 'Error: $e');
    }
  }

  // ---------- Galería: elegir y clasificar ----------
  Future<void> _pickAndClassify() async {
    final XFile? xf = await _picker.pickImage(source: ImageSource.gallery, maxWidth: 1024);
    if (xf == null) return;

    final bytes = await File(xf.path).readAsBytes();
    final savedPath = await _saveImageToGallery(xf);

    await _classifyBytesAndUpdate(bytes, savedPath: savedPath);
  }

  // ---------- Historial simple ----------
  void _openHistory() {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) {
      return Scaffold(
        appBar: AppBar(title: const Text('Historial')),
        body: ListView.builder(
          itemCount: _history.length,
          itemBuilder: (_, i) {
            final h = _history[i];
            return ListTile(
              leading: Image.file(File(h.path), width: 48, height: 48, fit: BoxFit.cover),
              title: Text('Contenedor ${h.label.toLowerCase()} '
                  '(${(h.conf * 100).toStringAsFixed(0)}%)'),
              subtitle: Text(DateFormat('yyyy-MM-dd HH:mm').format(h.ts)),
              onTap: () => showDialog(
                context: context,
                builder: (_) => Dialog(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Image.file(File(h.path)),
                      Padding(
                        padding: const EdgeInsets.all(12),
                        child: Text('Contenedor ${h.label.toLowerCase()} – '
                            '${(h.conf * 100).toStringAsFixed(0)}%'),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      );
    }));
  }

  // ---------- Util: reshape plano → [1, H, W, C] ----------
  List _reshape4D(Float32List flat, int b, int h, int w, int c) {
    int idx = 0;
    final out = List.generate(
      b,
          (_) => List.generate(
        h,
            (_) => List.generate(
          w,
              (_) => List.filled(c, 0.0),
        ),
      ),
    );
    for (int bi = 0; bi < b; bi++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          for (int ch = 0; ch < c; ch++) {
            out[bi][y][x][ch] = flat[idx++].toDouble();
          }
        }
      }
    }
    return out;
  }



  // -------------------- UI --------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ReciclAI'),
        actions: [
          IconButton(
            tooltip: _torchOn ? 'Apagar flash' : 'Encender flash',
            icon: Icon(_torchOn ? Icons.flash_on : Icons.flash_off),
            onPressed: _toggleTorch,
          ),
          IconButton(
            tooltip: 'Clasificar desde galería',
            icon: const Icon(Icons.photo_library),
            onPressed: _pickAndClassify,
          ),
          IconButton(
            tooltip: 'Historial',
            icon: const Icon(Icons.history),
            onPressed: _openHistory,
          ),
          IconButton(
            tooltip: 'Salir',
            icon: const Icon(Icons.close),
            onPressed: () => SystemNavigator.pop(),
          ),
        ],
      ),
      body: _isReady
          ? Stack(
        children: [
          CameraPreview(_controller!),
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 300),
              opacity: _bannerOpacity,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
                padding: const EdgeInsets.all(16),
                color: _barColor,
                child: Text(
                  _result == '—' ? 'Toca capturar' : _result,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
          ),
        ],
      )
          : const Center(child: CircularProgressIndicator()),
      floatingActionButton: _isReady
          ? FloatingActionButton.extended(
        onPressed: _captureAndClassify,
        icon: const Icon(Icons.camera),
        label: const Text('Capturar'),
      )
          : null,
    );
  }
}
