import tensorflow as tf

def convert_to_int8(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Post-Training Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]

    tflite_quant_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    convert_to_int8("mobilenet_v3.h5", "mobilenet_v3_int8.tflite")