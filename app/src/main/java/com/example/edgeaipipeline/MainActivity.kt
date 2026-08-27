package com.example.edgeaipipeline

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

class MainActivity : ComponentActivity() {

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize TFLite Interpreter with INT8 Model
        try {
            val modelBuffer = loadModelFile("mobilenet_v3_int8.tflite")
            interpreter = Interpreter(modelBuffer)
            Log.d("TFLite", "MobileNet INT8 Model Loaded Successfully!")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading model", e)
        }

        setContent {
            var inferenceTime by remember { mutableStateOf("Not executed yet") }
            var topResult by remember { mutableStateOf("--") }

            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "Edge AI Inference Pipeline",
                        style = MaterialTheme.typography.headlineMedium
                    )
                    Spacer(modifier = Modifier.height(20.dp))

                    Text(text = "Model: MobileNet V3 (INT8 Quantized)")
                    Spacer(modifier = Modifier.height(10.dp))

                    Text(text = "Inference Latency: $inferenceTime")
                    Spacer(modifier = Modifier.height(10.dp))

                    Text(text = "Output Tensor Sample: $topResult")
                    Spacer(modifier = Modifier.height(30.dp))

                    Button(onClick = {
                        val (latency, result) = runDummyInference()
                        inferenceTime = "$latency ms"
                        topResult = result
                    }) {
                        Text("Run Edge Inference Test")
                    }
                }
            }
        }
    }

    private fun runDummyInference(): Pair<Long, String> {
        val currentInterpreter = interpreter ?: return Pair(0L, "Interpreter Null")

        // MobileNet V3 standard input: 224x224 RGB image (UINT8 / INT8 = 1 byte per channel)
        val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        // 1001 output classes for ImageNet classification (INT8 array)
        val outputArray = Array(1) { ByteArray(1001) }

        // Measure execution latency in milliseconds
        val startTime = System.currentTimeMillis()
        currentInterpreter.run(inputBuffer, outputArray)
        val endTime = System.currentTimeMillis()

        val latency = endTime - startTime
        val sampleValue = outputArray[0][0].toString()

        return Pair(latency, "Class [0] raw score: $sampleValue")
    }

    @Throws(IOException::class)
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
private fun getInterpreterOptions(useGpu: Boolean): Interpreter.Options {
    val options = Interpreter.Options()
    if (useGpu) {
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = compatList.bestOptionsForThisDevice
            options.addDelegate(GpuDelegate(delegateOptions))
        }
    } else {
        options.setNumThreads(4) // Multithreading on CPU
    }
    return options
}
