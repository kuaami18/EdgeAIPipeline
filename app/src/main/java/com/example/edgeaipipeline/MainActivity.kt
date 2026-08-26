package com.example.edgeaipipeline

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            val modelBuffer = loadModelFile("model.tflite")
            interpreter = Interpreter(modelBuffer)
            Log.d("TFLite", "Interpreter created successfully!")
        } catch (e: Exception) {
            Log.e("TFLite", "Failed to load model", e)
        }
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