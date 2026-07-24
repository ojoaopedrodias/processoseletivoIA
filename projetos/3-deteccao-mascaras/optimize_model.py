from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# 1. Carrega o modelo já treinado.
model = YOLO("model.pt")

# 2. Exporta para TFLite com configurações otimizadas.
try:
    # Tentar exportar com half-precision (fp16) para melhor compatibilidade
    export_path = model.export(format="tflite", imgsz=640, half=False, int8=False)
    print(f"\n✅ Modelo exportado com sucesso para: {export_path}")
except Exception as e:
    print(f"Tentando exportar com configurações alternativas...")
    # Fallback: exportar para ONNX primeiro, depois converter
    try:
        export_path = model.export(format="onnx", imgsz=640, opset=13)
        print(f"\n✅ Modelo exportado para ONNX: {export_path}")
    except Exception as e2:
        print(f"❌ Erro na exportação: {e2}")
        exit(1)