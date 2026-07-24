import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# 1. Carrega o modelo YOLO11n já pré-treinado.
model = YOLO("yolo11n.pt")

# 2. Fine-tuning no dataset de máscaras, rodando em CPU.
results = model.train(
    data="dataset/data.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    device="cpu",
)

# 3. Copia o melhor checkpoint gerado durante o treino para model.pt, na raiz da pasta
best_weights = results.save_dir / "weights" / "best.pt"
shutil.copy(best_weights, "model.pt")

print(f"\n✅ Treinamento concluído. Pesos salvos em: model.pt (copiado de {best_weights})")
