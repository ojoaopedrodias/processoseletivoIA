# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** **João Pedro de Oliveira Dias**

### 1️⃣ Resumo da Abordagem

O modelo YOLO11n foi fine-tunado por 25 épocas no dataset `dataset/data.yaml`, com imagens redimensionadas para 640x640 e batch size de 8 (valor baixo escolhido por conta do treino em CPU). O treino foi feito em `device="cpu"`, conforme exigido.

Não foi aplicada nenhuma técnica específica de balanceamento de classes, o modelo foi treinado diretamente sobre a distribuição original do dataset. O efeito desse desbalanceamento ficou visível tanto nas métricas de treino (evolução mais lenta/instável) quanto na inferência final.

---
### 2️⃣ Bibliotecas Utilizadas

- `ultralytics` versão 8.4.104 (framework YOLO, treino, exportação e inferência)
- `torch` versão 2.13.0 / `torchvision` versão 0.28.0 (backend de deep learning)
dependências de exportação instaladas automaticamente pela Ultralytics no primeiro `optimize_model.py` (onnx, tensorflow, ai-edge-litert)

---
### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, o modelo treinado (`model.pt`) foi carregado e exportado para o formato TFLite através de `model.export(format="tflite", imgsz=640)`. Internamente, a Ultralytics converte o modelo em uma cadeia de formatos (PyTorch → ONNX → TensorFlow → TFLite) de forma automática, instalando as dependências necessárias na primeira execução. O resultado é o arquivo `model.tflite`, otimizado para inferência em dispositivos de borda (edge), aceitando uma imagem por vez (batch=1).

---
### 4️⃣ Resultados Obtidos

Na validação, o modelo atingiu:

mAP50 (época final, 25): 0.7663

mAP50-95 (época final, 25): 0.5308

Melhor mAP50 durante o treino: 0.7795 (época 22), com mAP50-95 de 0.5386

**Métricas por classe:**

| Classe | mAP50 |	mAP50-95 |
|:------ |:-----:| ---------:|
| with_mask | 0.9668 | 0.6761|
| without_mask | 0.8028 | 0.5253 |
| mask_weared_incorrect | 0.5694 | 0.4143 |

| Média (all) | 0.7796 | 0.5386 |
|:------ |:-----:| ------:|

O breakdown por classe confirma o padrão esperado pelo desbalanceamento do dataset: with_mask (classe majoritária) atinge desempenho excelente, without_mask vem em seguida, e mask_weared_incorrect (classe minoritária) fica com o resultado mais baixo — ainda assim, um valor de 0.5694 é razoável para uma classe com poucos exemplos de treino, mostrando que o modelo conseguiu aprender um padrão real, mesmo que menos robusto que nas outras classes.

**Tamanho dos arquivos:**

`model.pt`: 5.3 MB

`model.tflite`: 11 MB

---

### 5️⃣ Comentários Adicionais (Opcional)

O maior desafio do projeto não foi o modelo em si, mas o ambiente de execução. Foram necessários ajustes de Dev Container/Docker, incluindo resolver um erro de arquitetura na exportação TFLite e realinhar as versões de torch/torchvision que ficaram incompatíveis após a instalação das dependências de exportação. De modo geral, as maiores dficuldades não foram fazer o que o projeto pede, mas sim tentar corrigir os erros que aparecem antes de começar o projeto e na hora de testa-lo.

### 6️⃣ Exemplo de Inferência

---
Projeto 3 — Inferência com model.tflite (Edge AI)
---

Rodando inferência em 5 amostras usando model.tflite:

| Imagem | Detecções | Detalhes |
|:--- |:-----:| -----:|
| maksssksksss105.jpg | 9 | [9x with_mask] |
| maksssksksss107.jpg | 1 | [1x with_mask] |
|maksssksksss11.jpg | 25 | [24x with_mask, 1x mask_weared_incorrect] |
| maksssksksss113.jpg | 4 | [3x with_mask, 1x without_mask] |
| maksssksksss12.jpg | 16 | [13x with_mask, 3x without_mask] |

| TOTAL | 55 |
|:--- | ---:|

---
Ao abrir as imagens anotadas em `runs/detect/inferencia_exemplos/predicoes/`, observa-se que o modelo detecta com boa consistência a classe majoritária (with_mask), inclusive em imagens com múltiplos rostos pequenos (25 detecções na maksssksksss11.jpg). A classe without_mask também aparece de forma coerente. Já a classe mask_weared_incorrect praticamente não é detectada, apenas 1 ocorrência em todas as 5 imagens. Além disso, juntamente com as caixas há também as classes, o que, em alguns casos, dificutou a visualização de outras caixas.

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
