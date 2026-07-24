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

# 2. Exporta para TFLite.
export_path = model.export(format="tflite", imgsz=640)

print(f"\n✅ Modelo exportado com sucesso para: {export_path}")
