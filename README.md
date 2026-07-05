# Classificação de categorias de imagens a partir de fMRI (NSD)

Trabalho final de **Machine Learning 2 (IMPA)** — decodificação neural: dado o padrão
de atividade cerebral (fMRI) de um sujeito enquanto ele observava uma imagem natural,
prever **a que categoria de conteúdo a imagem pertence**.

## Apresentação e objetivos

Usamos o **Natural Scenes Dataset (NSD)** — respostas fMRI de alta resolução do sujeito
`subj01` a milhares de imagens do COCO. A partir do conteúdo COCO de cada imagem vista,
derivamos um problema de classificação com **3 classes**:

| rótulo | classe   |
|--------|----------|
| `0`    | Pessoa   |
| `1`    | Animal   |
| `2`    | Outro    |

O objetivo é comparar, de forma controlada, **como a representação dos dados** e **a
arquitetura do modelo** afetam a capacidade de decodificar a categoria a partir do sinal
cerebral. Estudamos duas dimensões cruzadas:

- **Representação do sinal**
  - **Volumétrica** — o volume 3D inteiro do cérebro `(83,104,81)` reamostrado para
    `(91,109,91)`, consumido por **CNNs 3D**.
  - **ROI (`nsdgeneral`)** — apenas os voxels do córtex visual mascarados pela ROI
    `nsdgeneral`, achatados em um vetor de ~15.724 voxels, consumido por **MLPs**.
- **Agregação dos trials**
  - **AVG** — média das repetições da mesma imagem (menos ruído, menos exemplos).
  - **IND** — cada trial individual como um exemplo (mais dados, mais ruído).

Isso gera a matriz de comparação **(AVG / IND) × (Volumétrico / ROI)**.

> **Fluxo:** os notebooks de pré-processamento leem os betas brutos do NSD e escrevem
> datasets em disco; `main.ipynb` treina e avalia a partir desses arquivos. Não há import
> compartilhado entre as duas etapas — a comunicação é apenas por arquivos.

---

## Downloads dos datasets

Os betas do NSD e as anotações do COCO **não** estão versionados no repositório (são
dezenas de GB). Baixe-os para os caminhos abaixo antes de rodar o pré-processamento.

Os notebooks esperam:

```
~/Downloads/dataset/nsd/                      # dados brutos do NSD (betas, expdesign)
~/Downloads/dataset/data_nsd/processed/       # saída dos notebooks de pré-processamento
```

### 1. Natural Scenes Dataset (NSD) — betas + design experimental

O NSD é distribuído em um bucket público na AWS S3 (`natural-scenes-dataset`). Requer
[AWS CLI](https://aws.amazon.com/cli/) (`--no-sign-request`, sem credenciais).

```bash
NSD=~/Downloads/dataset/nsd
mkdir -p $NSD/subj01/betas

# Design experimental (mapeia trial -> imagem)  [pequeno, ~alguns MB]
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat \
  $NSD/nsd_expdesign.mat

# Betas do subj01 (func1pt8mm, GLMsingle) — 1 arquivo por sessão (~1.9 GB cada, 40 sessões)
# Baixe quantas sessões quiser; o pipeline itera sobre betas_session*.hdf5 encontrados.
for s in $(seq -w 1 40); do
  aws s3 cp --no-sign-request \
    s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session${s}.hdf5 \
    $NSD/subj01/betas/betas_session${s}.hdf5
done
```

### 2. Anotações e imagens do COCO 2017 (para rotular as imagens)

```bash
AUX=trabalho_4/cnn_final/data_loader/auxiliar
mkdir -p $AUX

# Anotações de instâncias (categorias/áreas por imagem) — usadas para gerar os rótulos
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o /tmp/ann.zip
unzip -j /tmp/ann.zip "annotations/instances_train2017.json" -d $AUX

# Imagens de treino 2017 (~18 GB) — usadas só na visualização/inspeção
curl -L http://images.cocodataset.org/zips/train2017.zip -o /tmp/train2017.zip
unzip -q /tmp/train2017.zip -d $AUX     # cria $AUX/train2017/
```

### 3. Metadados de estímulos e máscara ROI do NSD (arquivos auxiliares)

```bash
AUX=trabalho_4/cnn_final/data_loader/auxiliar

# Info dos estímulos NSD (mapeia nsdId <-> coco_id)
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl \
  $AUX/nsd_stim_info_merged.pkl

# Máscara nsdgeneral do subj01 (ROI do córtex visual, usada no pipeline ROI)
mkdir -p $AUX/nsddata/ppdata/subj01/func1pt8mm/roi
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz \
  $AUX/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz
```

### 4. Pesos pré-treinados do MedicalNet (backbones 3D ResNet)

Baixe `resnet_10.pth` e `resnet_18.pth` do repositório oficial
[Tencent/MedicalNet](https://github.com/Tencent/MedicalNet) (link Google Drive/OneDrive na
seção *"Pretrain models"*) e coloque em `data_loader/auxiliar/`.

---

## Arquivos em `data_loader/auxiliar/`

Depois dos downloads, a pasta deve conter (nem tudo é versionado):

| Arquivo | Origem | Uso |
|---|---|---|
| `nsd_stim_info_merged.pkl` / `.csv` | NSD (`nsddata/experiments`) | Mapeia `nsdId` (0-indexed) → `coco_id`. **Cuidado:** `subjectim` do `expdesign` é 1-indexed. |
| `instances_train2017.json` | COCO 2017 | Categorias/áreas por imagem → rótulos (Pessoa/Animal/Outro). |
| `instances_train2014.json` | COCO 2014 | Compat. com imagens NSD que referenciam o split 2014. |
| `train2017/` | COCO 2017 | Imagens JPG, usadas apenas para inspeção/visualização. |
| `nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz` | NSD | Máscara ROI do córtex visual (pipeline ROI). |
| `nsdgeneral.nii` | derivado | Versão descompactada da máscara. |
| `resnet_10.pth`, `resnet_18.pth` | MedicalNet | Pesos pré-treinados dos backbones 3D ResNet-10/18. |

Os caminhos estão fixados no topo de `process_data_NSD.ipynb` e `process_data_ROI.ipynb`
(`STIM_INFO`, `COCO_ANNOTATIONS`, `MASKS_DIR`, etc.) — ajuste se mudar de máquina.

---

## Explicação dos modelos

Fábrica única em [model.py](model.py) — `create_model(model_name, dropout, dataset, cfg)`
— que **ramifica no `model_name`** (não no dataset). Otimizadores em
[optimizer.py](optimizer.py): `adam`, `adam_cosine`, `sgd`, `sgd_plateau`.

Todas as CNNs 3D usam um **pooling `AvgMaxPool3d`** (concatena average + max pool 3D)
para preservar mais informação espacial antes do classificador.

### CNNs 3D (representação volumétrica)

| `model_name` | Descrição | Classificador |
|---|---|---|
| `shallowcnn` | 2 blocos conv (16→32) com BatchNorm; leve, rápido. | 512 → 128 → 3 |
| `cnn`        | Baseline: 3 convs com stride (16→32→64), sem BN. | 1024 → 256 → 3 |
| `deepcnn`    | 7 convs com BatchNorm e MaxPool (16→128); mais capacidade. | 2048 → 512 → 3 |
| `medicalnet` | Backbone **3D ResNet-10/18** do MedicalNet (pré-treinado em imagens médicas), com `conv1` adaptada para 1 canal e cabeça linear nova; opção de congelar o backbone. | 8192 → 512 → 3 |

### MLPs (representação ROI, vetor flat ~15.7k voxels)

Exigem `cfg['n_voxels']` (lido do HDF5).

| `model_name` | Arquitetura |
|---|---|
| `mlp_small`  | `n_voxels → 512 → 128 → 3` (BatchNorm + Dropout). |
| `mlp_medium` | `n_voxels → 2048 → 512 → 128 → 3`. |
| `mlp_large`  | `n_voxels → 4096 → [res 1024] → [res 256] → 3` com conexões residuais. |

### Mapeamento dataset → dataset class → família de modelo

| `dataset` (config) | Arquivo de dados | Dataset class | Modelos |
|---|---|---|---|
| `nsd_complete_AVG` | `AVG/averaged_betas.hdf5` | `fMRIAVGDataset` | CNN 3D |
| `nsd_complete_IND` | `IND/*.p` | `fMRICNNcustomDataset` | CNN 3D |
| `nsd_complete_ROI` | `ROI/roi_avg_betas.hdf5` | `fMRIROIDataset` | MLP |
| `nsd_complete_ROI_IND` | `ROI_IND/roi_ind_betas.hdf5` | `fMRIROIDataset` | MLP |

Estratégias de treino avaliadas (sufixos nos nomes dos experimentos): `class_weight`
(`_cw`), `focal loss` (`_focal`/`_g4`), dropout (`_drop05`), cosine annealing (`_cosine`),
regularização/weight decay (`_reg`, `_wd`) e augmentation (`_aug`).

---

## Resultados finais

Métrica principal: **macro-F1** na validação (melhor época). Os números abaixo são o melhor
experimento por variante. O split treino/teste é feito por `coco_id` (nível de imagem) para
evitar vazamento; a **métrica honesta** é o conjunto de teste held-out avaliado por
`visualize_model_predictions` em `main.ipynb`.

| Variante | Representação | Melhor modelo | macro-F1 | acc |
|---|---|---|---:|---:|
| **NSD ROI (AVG)**  | ROI flat, MLP | `mlp_small_adam`        | **0.910** | 0.914 |
| NSD ROI (IND)      | ROI flat, MLP | `mlp_large_cosine_cw`   | 0.861 | 0.862 |
| NSD ROI (COMPLETE) | ROI flat, MLP | `mlp_small_reg_cw`      | 0.863 | 0.867 |
| **NSD Volumétrico (AVG)** | Volume 3D, CNN | `deepcnn_focal`   | **0.776** | 0.799 |
| NSD Volumétrico (COMPLETE)| Volume 3D, CNN | `cnn_cosine_cw`   | 0.724 | 0.752 |
| NSD Volumétrico (IND)     | Volume 3D, CNN | `deepcnn_cosine_cw` | 0.340 | 0.343 |
| BOLD5000 (referência)     | Volume 3D, CNN | `deepcnn_lr1e4_cw`  | 0.420 | 0.433 |

### Conclusões

- **A representação ROI supera a volumétrica com folga** (~0.91 vs ~0.78 macro-F1 no melhor
  caso AVG). Restringir aos voxels do córtex visual (`nsdgeneral`) remove ruído e reduz
  drasticamente a dimensionalidade, facilitando o aprendizado.
- **Agregar por imagem (AVG) ajuda muito**: a média das repetições reduz o ruído do fMRI.
  Nas variantes **IND** (trial a trial), as CNNs 3D ficam praticamente no acaso (~0.34, com
  3 classes o chance é ~0.33), enquanto os MLPs sobre ROI ainda decodificam bem (~0.86).
- **Na ROI, a tarefa é limitada por dados, não por modelo**: `mlp_small`, `medium` e `large`
  se agrupam em ~0.88–0.91, e um decodificador linear atinge desempenho comparável. Ajustar
  arquitetura/otimizador move pouco — os ganhos vêm do lado dos dados (ROIs mais específicas,
  seleção de voxels, mais sujeitos/sessões).
- **MedicalNet congelado não transfere** para betas de fMRI (colapsa próximo do chance);
  só o fine-tuning parcial (`_finetune`) chega perto das CNNs treinadas do zero.

> Rótulos entre pipelines **não são estritamente comparáveis**: o pipeline volumétrico usa
> prioridade simples pessoa>animal>outro, enquanto o ROI usa frações de área mínimas
> (`PERSON_MIN_FRAC=0.05`, `ANIMAL_MIN_FRAC=0.02`) e restringe *Outro* a supercategorias
> específicas (`vehicle`, `furniture`, `outdoor`).
