# Classificação de categorias de imagens a partir de fMRI

Trabalho final de **Machine Learning 2 (IMPA)** — decodificação neural: dado o padrão
de atividade cerebral (fMRI) de um sujeito enquanto observava uma imagem natural,
prever **a que categoria semântica a imagem pertence**.

O projeto também integra uma iniciação científica no **IDOR**, cujo objetivo é
quantificar numericamente a discrepância de desempenho entre equipamentos de
**7 Teslas** (padrão ouro, maior SNR) e **3 Teslas** (equipamento disponível no IDOR)
em tarefas de decodificação neural.

---

## Visão geral

O problema é uma **classificação com 3 classes** derivada das anotações do COCO:

| Rótulo | Classe  | Critério de área mínima |
|--------|---------|------------------------|
| `0`    | Pessoa  | ≥ 5 % da imagem        |
| `1`    | Animal  | ≥ 2 % da imagem        |
| `2`    | Outro   | supercategorias `vehicle`, `furniture`, `outdoor` |

Imagens que não atendem a nenhum critério são descartadas.

O estudo cruza duas dimensões para comparação controlada:

- **Representação do sinal:** volume cerebral 3D inteiro (*volumétrico*) vs. apenas os
  voxels do córtex visual (*ROI nsdgeneral*, vetor flat ~15.7 k voxels).
- **Agregação temporal:** média das repetições da mesma imagem (*AVG*, menos ruído) vs.
  cada trial individual (*IND*, mais dados porém mais ruidoso).

Os datasets usados são o **NSD** (`subj01`, 7T) e o **BOLD5000** (`CSI1`, 3T).

> **Fluxo geral:** os notebooks de pré-processamento leem os betas brutos e escrevem
> datasets em disco; `main.ipynb` treina e avalia a partir desses arquivos. Não há
> import compartilhado entre as duas etapas — a comunicação é feita apenas por arquivos.

---

## Estrutura do repositório

```
.
├── process_data_NSD.ipynb       # Pré-processamento volumétrico (AVG e IND)
├── process_data_ROI.ipynb       # Pré-processamento ROI (ROI-AVG, ROI-IND, ROI-AUG)
├── process_data_bold.ipynb      # Pré-processamento BOLD5000
├── main.ipynb                   # Treinamento e avaliação
├── model.py                     # Fábrica de modelos (CNN3D e MLP)
├── optimizer.py                 # Otimizadores (Adam, SGD + schedulers)
└── data_loader/
    └── auxiliar/                # Arquivos auxiliares (ver seção de downloads)
```

---

## Setup

```bash
# Criar ambiente virtual e instalar dependências
python3 -m venv venv_nsd
source venv_nsd/bin/activate
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118   # ou /cpu se não tiver GPU
pip install numpy scipy h5py nibabel nilearn \
            pandas scikit-learn matplotlib tqdm pycocotools jupyter
```

---

## Downloads

Os betas do NSD e as anotações do COCO **não estão versionados** no repositório (são
dezenas de GB). Baixe-os antes de rodar os notebooks de pré-processamento.

### 1. NSD — betas e design experimental (7T)

O NSD é distribuído em um bucket público na AWS S3. Requer
[AWS CLI](https://aws.amazon.com/cli/) — sem necessidade de credenciais.

```bash
NSD=~/Downloads/dataset/nsd
mkdir -p $NSD/subj01/betas

# Design experimental (trial → imagem) — poucos MB
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat \
  $NSD/nsd_expdesign.mat

# Betas do subj01 — ~1.9 GB por sessão, 40 sessões no total
# Passe um número menor para testar (ex: seq 1 3 para as 3 primeiras sessões)
for s in $(seq -w 1 40); do
  aws s3 cp --no-sign-request \
    s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session${s}.hdf5 \
    $NSD/subj01/betas/betas_session${s}.hdf5
done
```

### 2. COCO 2017 — anotações e imagens

```bash
AUX=data_loader/auxiliar
mkdir -p $AUX

# Anotações de instâncias (necessário para gerar os rótulos)
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
     -o /tmp/ann.zip
unzip -j /tmp/ann.zip "annotations/instances_train2017.json" -d $AUX

# Imagens de treino (~18 GB) — necessário apenas para inspeção/visualização
curl -L http://images.cocodataset.org/zips/train2017.zip -o /tmp/train2017.zip
unzip -q /tmp/train2017.zip -d $AUX   # cria $AUX/train2017/
```

### 3. Metadados e máscara ROI do NSD

```bash
AUX=data_loader/auxiliar

# Mapeamento nsdId ↔ coco_id
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl \
  $AUX/nsd_stim_info_merged.pkl

# Máscara nsdgeneral do subj01 (córtex visual — usada no pipeline ROI)
mkdir -p $AUX/nsddata/ppdata/subj01/func1pt8mm/roi
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz \
  $AUX/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz
```

### 4. Pesos pré-treinados do MedicalNet (ResNet 3D)

Baixe `resnet_10.pth` e `resnet_18.pth` do repositório oficial
[Tencent/MedicalNet](https://github.com/Tencent/MedicalNet) (seção *"Pretrain models"*)
e coloque em `data_loader/auxiliar/`.

### Estrutura esperada em `data_loader/auxiliar/`

| Arquivo | Origem | Uso |
|---|---|---|
| `nsd_stim_info_merged.pkl` | NSD S3 | Mapeia `nsdId` (0-indexed) → `coco_id` |
| `instances_train2017.json` | COCO 2017 | Rótulos por imagem (Pessoa / Animal / Outro) |
| `instances_train2014.json` | COCO 2014 | Compatibilidade com referências 2014 no NSD |
| `train2017/` | COCO 2017 | Imagens JPG (só para visualização) |
| `nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz` | NSD S3 | Máscara ROI |
| `resnet_10.pth`, `resnet_18.pth` | MedicalNet | Pesos pré-treinados dos backbones 3D |

> ⚠️ Os caminhos estão fixados no topo de cada notebook de pré-processamento
> (`STIM_INFO`, `COCO_ANNOTATIONS`, `MASKS_DIR`, etc.). Ajuste-os se mudar de máquina.

---

## Pré-processamento

Rode os notebooks na ordem abaixo. Cada um escreve seus datasets em disco; `main.ipynb`
lê a partir dessas pastas.

| Notebook | Pipelines gerados | Formato de saída |
|---|---|---|
| `process_data_NSD.ipynb` | `AVG` (média por imagem), `IND` (por trial) | `.hdf5` / `.p` — tensor `(1, 91, 109, 91)` |
| `process_data_ROI.ipynb` | `ROI-AVG`, `ROI-IND`, `ROI-AUG` (+ ruído gaussiano) | `.hdf5` — vetor `(~15 724,)` |
| `process_data_bold.ipynb` | `BOLD` (por trial, 5 TRs) | `.p` — tensor `(5, 91, 109, 91)` |

**Notas importantes de pré-processamento:**

- Os betas brutos têm eixos transpostos em relação à máscara NIfTI — o código aplica
  `.transpose(2, 1, 0)` antes de usar a máscara ROI.
- Os índices em `nsd_expdesign.mat` são **1-based**; o dicionário `nsdId → cocoId` é
  **0-based**. Ambas as conversões são aplicadas no código (`mo - 1` e `subjectim[...] - 1`).
- Voxels `NaN` são substituídos por zero (`nan_to_num`), processados por trial para
  evitar pico de memória (~11 GB por sessão inteira).
- O split treino/teste é feito por `coco_id` (nível de imagem) para evitar vazamento
  entre repetições da mesma imagem.

---

## Modelos

Fábrica única em `model.py` — `create_model(model_name, dropout, dataset, cfg)`.
Otimizadores em `optimizer.py`: `adam`, `adam_cosine`, `sgd`, `sgd_plateau`.

Todas as CNNs 3D usam **`AvgMaxPool3d`** (concatena average + max pooling 3D) antes do
classificador para preservar mais informação espacial.

### CNNs 3D — representação volumétrica

| `model_name` | Descrição | Classificador |
|---|---|---|
| `shallowcnn` | 2 blocos conv (16 → 32) com BatchNorm | 512 → 128 → 3 |
| `cnn`        | 3 convs com stride (16 → 32 → 64), sem BN | 1024 → 256 → 3 |
| `deepcnn`    | 7 convs com BatchNorm e MaxPool (16 → 128) | 2048 → 512 → 3 |
| `medicalnet` | Backbone **3D ResNet-10/18** (MedicalNet), `conv1` adaptada para 1 canal; fine-tuning parcial disponível | 8192 → 512 → 3 |

### MLPs — representação ROI (vetor flat ~15.7 k voxels)

Exigem `cfg['n_voxels']` (lido automaticamente do HDF5).

| `model_name`  | Arquitetura |
|---|---|
| `mlp_small`   | `n_voxels → 512 → 128 → 3` (BatchNorm + Dropout) |
| `mlp_medium`  | `n_voxels → 2048 → 512 → 128 → 3` |
| `mlp_large`   | `n_voxels → 4096 → [res 1024] → [res 256] → 3` com conexões residuais |

### Mapeamento dataset → classe de dataset → modelos

| `dataset` (config) | Arquivo de dados | Dataset class | Modelos compatíveis |
|---|---|---|---|
| `nsd_complete_AVG`     | `AVG/averaged_betas.hdf5`     | `fMRIAVGDataset`       | CNN 3D |
| `nsd_complete_IND`     | `IND/*.p`                     | `fMRICNNcustomDataset` | CNN 3D |
| `nsd_complete_ROI`     | `ROI/roi_avg_betas.hdf5`      | `fMRIROIDataset`       | MLP    |
| `nsd_complete_ROI_IND` | `ROI_IND/roi_ind_betas.hdf5`  | `fMRIROIDataset`       | MLP    |

Sufixos usados nos nomes de experimento: `_cw` (class weight), `_focal` / `_g4` (focal
loss), `_drop05` (dropout), `_cosine` (cosine annealing), `_reg` / `_wd` (weight decay),
`_aug` (augmentation com ruído gaussiano).

---

## Resultados

Métrica principal: **macro-F1** na validação (melhor época). O split treino/teste é por
`coco_id` para evitar vazamento. A métrica definitiva é avaliada no conjunto de teste
held-out via `visualize_model_predictions` em `main.ipynb`.

| Variante | Representação | Melhor modelo | macro-F1 | Acurácia |
|---|---|---|---:|---:|
| **NSD ROI (AVG)**         | ROI flat · MLP  | `mlp_small_adam`          | **0.910** | 0.914 |
| NSD ROI (IND)             | ROI flat · MLP  | `mlp_large_cosine_cw`     | 0.861     | 0.862 |
| NSD ROI (COMPLETE)        | ROI flat · MLP  | `mlp_small_reg_cw`        | 0.863     | 0.867 |
| **NSD Volumétrico (AVG)** | Volume 3D · CNN | `deepcnn_focal`           | **0.776** | 0.799 |
| NSD Volumétrico (COMPLETE)| Volume 3D · CNN | `cnn_cosine_cw`           | 0.724     | 0.752 |
| NSD Volumétrico (IND)     | Volume 3D · CNN | `deepcnn_cosine_cw`       | 0.340     | 0.343 |
| BOLD5000 (referência 3T)  | Volume 3D · CNN | `deepcnn_lr1e4_cw`        | 0.420     | 0.433 |

### Principais conclusões

- **ROI supera volumétrico com folga** (~0.91 vs ~0.78 macro-F1). Restringir a
  entrada ao córtex visual remove ruído e reduz drasticamente a dimensionalidade.
- **AVG ajuda muito**: a média das repetições reduz o ruído do fMRI. Com trials
  individuais (IND), as CNNs 3D colapsam próximo ao acaso (~0.34; chance = 0.33),
  enquanto os MLPs sobre ROI ainda decodificam bem (~0.86).
- **Na ROI, o gargalo são os dados, não o modelo**: `mlp_small`, `medium` e `large`
  ficam todos em ~0.88–0.91, e um decodificador linear atinge desempenho comparável.
  Os ganhos vêm do lado dos dados — ROIs mais específicas, seleção de voxels, mais
  sujeitos ou sessões.
- **MedicalNet congelado não transfere** para betas de fMRI (colapsa no acaso). Apenas
  o fine-tuning parcial (`_finetune`) chega perto das CNNs treinadas do zero.
- **BOLD5000 (3T) vs NSD (7T)**: o modelo não conseguiu aprender padrões consistentes
  no BOLD5000, mesmo equiparando o volume de dados. A discrepância de qualidade de
  sinal entre 3T e 7T tem impacto expressivo na decodificação.

> ⚠️ Os rótulos entre pipelines **não são estritamente comparáveis**: o pipeline
> volumétrico usa prioridade simples pessoa > animal > outro, enquanto o ROI usa
> frações de área mínimas (`PERSON_MIN_FRAC = 0.05`, `ANIMAL_MIN_FRAC = 0.02`) e
> restringe *Outro* a supercategorias específicas.

---

## Referências

- [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/)
- [BOLD5000](https://bold5000-dataset.github.io/website/)
- [MedicalNet — Tencent](https://github.com/Tencent/MedicalNet)
- Repositório base: [harsharaman/bold5000_fmri](https://github.com/harsharaman/bold5000_fmri)
