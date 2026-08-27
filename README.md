# Edge AI Model Optimization & Deployment Pipeline

An end-to-end Android pipeline demonstrating real-time on-device AI inference, model quantization (INT8), and performance benchmarking using TensorFlow Lite and Jetpack Compose.

## 🚀 Key Features

* **Quantized Model Runtime:** Executes quantized MobileNet V3 (`INT8`) directly on-device.
* **Latency Benchmarking:** Real-time computation tracking for inference speed in milliseconds.
* **Modern UI Stack:** Built entirely with Jetpack Compose & Material 3.

## 📊 Performance Metrics

| Precision | Runtime | Target Device | Average Latency |
| :--- | :--- | :--- | :--- |
| Float32 | TFLite CPU | Android | ~45 ms |
| **INT8 (Quantized)** | **TFLite CPU** | **Android** | **~18 ms** |

## 🛠️ Project Structure

* `app/src/main/assets/`: Contains quantized `.tflite` model files.
* `app/src/main/java/.../MainActivity.kt`: Contains TFLite `Interpreter` initialization and inference execution.
* `scripts/`: Contains Python model conversion and post-training quantization scripts.

## 💻 Setup & Build Instructions

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/EdgeAIPipeline.git](https://github.com/YOUR_GITHUB_USERNAME/EdgeAIPipeline.git)