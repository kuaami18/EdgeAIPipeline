package com.example.edgeaipipeline

import android.Manifest
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            val modelBuffer = loadModelFile("mobilenet_v3_int8.tflite")
            interpreter = Interpreter(modelBuffer)
            Log.d("TFLite", "MobileNet INT8 Model Loaded Successfully!")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading model", e)
        }

        setContent {
            Surface(modifier = Modifier.fillMaxSize()) {
                CameraPermissionWrapper {
                    LiveCameraInferenceScreen(interpreter)
                }
            }
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraPermissionWrapper(content: @Composable () -> Unit) {
    val permissionState = rememberPermissionState(Manifest.permission.CAMERA)

    LaunchedEffect(Unit) {
        if (!permissionState.status.isGranted) {
            permissionState.launchPermissionRequest()
        }
    }

    if (permissionState.status.isGranted) {
        content()
    } else {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission required for live edge inference.")
        }
    }
}

@Composable
fun LiveCameraInferenceScreen(interpreter: Interpreter?) {
    var latencyMs by remember { mutableLongStateOf(0L) }
    var classIndex by remember { mutableStateOf(0) }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera Preview Feed
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(224, 224))
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { analyzer ->
                            analyzer.setAnalyzer(ContextCompat.getMainExecutor(ctx)) { imageProxy ->
                                processFrame(imageProxy, interpreter) { latency, topClass ->
                                    latencyMs = latency
                                    classIndex = topClass
                                }
                            }
                        }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalyzer
                        )
                    } catch (e: Exception) {
                        Log.e("CameraX", "Binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Benchmarking Overlay UI
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.7f))
                .padding(16.dp)
        ) {
            Text(text = "Model: MobileNet V3 (INT8 Quantized)", color = Color.White)
            Text(text = "Live Latency: $latencyMs ms", color = Color.Green)
            Text(text = "Top Predicted Class ID: $classIndex", color = Color.White)
        }
    }
}

private fun processFrame(
    imageProxy: ImageProxy,
    interpreter: Interpreter?,
    onResult: (Long, Int) -> Unit
) {
    if (interpreter == null) {
        imageProxy.close()
        return
    }

    val bitmap = imageProxy.toBitmap()
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

    // Convert RGBA Bitmap into 224x224x3 INT8 Direct ByteBuffer for MobileNet
    val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3).apply {
        order(ByteOrder.nativeOrder())
    }

    val intValues = IntArray(224 * 224)
    resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)

    for (pixel in intValues) {
        val r = ((pixel shr 16) and 0xFF) - 128
        val g = ((pixel shr 8) and 0xFF) - 128
        val b = (pixel and 0xFF) - 128
        inputBuffer.put(r.toByte())
        inputBuffer.put(g.toByte())
        inputBuffer.put(b.toByte())
    }

    val outputArray = Array(1) { ByteArray(1001) }

    val startTime = System.currentTimeMillis()
    interpreter.run(inputBuffer, outputArray)
    val endTime = System.currentTimeMillis()

    // Extract ArgMax Class ID
    var maxIdx = 0
    var maxVal = outputArray[0][0]
    for (i in outputArray[0].indices) {
        if (outputArray[0][i] > maxVal) {
            maxVal = outputArray[0][i]
            maxIdx = i
        }
    }

    onResult(endTime - startTime, maxIdx)
    imageProxy.close()
}