# Dados

Os dados ficam em `data/` (ignorada pelo git). Os dois conjuntos abaixo são **CC BY 4.0**: citar ao usar.

| Dataset | Origem | O que é | Como entra no projeto |
|---|---|---|---|
| **CSB-1** (WormSwin) | [Zenodo 7456803](https://doi.org/10.5281/zenodo.7456803) — Deserno & Bozek, 2023; artigo: *WormSwin: instance segmentation of C. elegans using vision transformer*, Sci. Rep. Vídeos fornecidos por M. Rieckher; máscaras feitas sob supervisão de Bozek e Deserno | **Vídeo** de placa (912×736, cinza gravado como RGB), uma máscara por verme, com sobreposição | Direto: `--dataset-root data/csb-1_extracted/csb-1_dataset --seq <seq>`. É o conjunto usado nos resultados |
| **Annotated images** | [Zenodo 15526615](https://doi.org/10.5281/zenodo.15526615) — Weheliye, Rodriguez, Feriani, Javer, Uhlmann & Brown, 2025; artigo: *A neural network model enables worm tracking in challenging conditions…*, PLOS Comput. Biol. 21(8): e1013345 | **Recortes independentes** (33–350 px) com 1–6 vermes; contorno e esqueleto por verme | Via `tools/convert_annotated_images.py` (abaixo). Serve para avaliar a segmentação; não é vídeo |

O `csb-1_dataset.zip` tem 10 sequências (`24_0_0_1` … `24_2_1_2`), `images/<seq>/<k>.png` e `masks/<seq>/<k>_<i>.png` (uma máscara por verme). O zip também traz `coco_annotations/`, que não é usado. A sequência `24_2_1_1` é a de **teste**; nenhuma ferramenta a usa para treinar.

Layout esperado (ajuste `input.pattern`/`input.mask_pattern` se os nomes forem outros):

```
data/csb-1_extracted/csb-1_dataset/
├── images/<seq>/<k>.png
└── masks/<seq>/<k>_<i>.png
```

Do registro 15526615 só se usa o `Annotated_images.zip` (1,6 GB); os demais (`DT_C`, `PAF`, `Omnipose`, `Figures_Code`, 38,9 GB) são modelos e código do artigo.

## Annotated images: conversão e uso

O zip contém 13 arquivos `<fonte>_<split>.p.zip` que, na verdade, são **GZIP de pickles** (fontes: `manual`, `manual-v2`, `from_NNv1`, `from_tierpsy`). Como pickle pode executar código, o conversor usa um desserializador que só aceita as classes do NumPy presentes nesses arquivos e recusa o resto.

```bash
# extrair o zip em data/annotated_raw e converter os conjuntos de teste
python -I tools/convert_annotated_images.py --src data/annotated_raw/Annotated_images \
    --out data/annotated_converted --only manual-v2_test manual_test from_NNv1_test from_tierpsy_test --skeletons

# segmentar (recortes independentes: fundo por top-hat, moldura preta ignorada)
python run_pdi.py --dataset-root data/annotated_converted --seq manual-v2_test --out runs/ann_manual-v2_test \
    --set background.method=tophat --set input.nodata_border=true \
    --set "stages=[normalize,denoise,background,threshold,morphology,labeling,analysis,split,refine,evaluate,export,visualize]"
```

- Cada amostra: imagem em cinza, larguras (49 pts), esqueleto (49 pts) e contorno (98 pts) por verme. As **máscaras são o contorno preenchido** (uma por verme, sobreposição preservada). A ordem dos eixos do contorno é detectada automaticamente (contraste local) e foi conferida visualmente nos 4 conjuntos.
- `from_NNv1_*` **não traz contorno**: as máscaras são **reconstruídas** do esqueleto + largura (IoU médio ≈ 0,88 contra o contorno anotado, medido em `manual-v2_test`). Tratar como gabarito aproximado.
- Não convertidos: os `*_train`/`*_validation` (o `from_tierpsy_train` tem 1,2 GB compactado) e `from_tierpsy_negative` (formato diferente, recortes sem vermes).
- `input.nodata_border=true`: os recortes têm moldura preta (valor 0) mais escura que o verme, que virava falso positivo; regiões de valor 0 ligadas à borda passam a ser "sem dado".

Resultado nos conjuntos de **teste** (só PDI, sem CNN; média por recorte; F1 por verme a IoU ≥ 0,5). Estes números são de uma versão anterior do pipeline (antes do ajuste do fundo e do crescimento da máscara) e **não foram refeitos**:

| Conjunto | vermes/recorte | F1 | recall em amontoados |
|---|---|---|---|
| `from_tierpsy_test` (1 verme, recortes fáceis) | 1,0 | 0,93 | 0,79 |
| `from_NNv1_test` (máscara reconstruída) | 1,65 | 0,71 | 0,52 |
| `manual_test` | 1,3 | 0,52 | 0,44 |
| `manual-v2_test` (mais sobreposição) | 2,1 | 0,44 | 0,24 |

O padrão é o esperado: o PDI vai bem em verme isolado e piora quando aumenta a sobreposição. Os parâmetros em pixels (`min_area`, `marker_min_dist`) estão nos valores do CSB-1, e a escala dos recortes varia muito (33 a 350 px).

## Recortes para a CNN (`stages/export.py`)

Cada amontoado vira um recorte `patch_size × patch_size` (padrão 192) centrado nele, gravado em `<saída>/cnn_dataset/<seq>/f<quadro>_c<componente>.npz`:

| Campo | Conteúdo |
|---|---|
| `image`, `signal` | quadro em cinza e sinal pós-fundo (uint8) |
| `mask` | máscara binária do amontoado (só aquele componente) |
| `dist`, `skeleton`, `ws_labels` | transformada de distância, esqueleto e proposta do watershed (a CNN **não** os usa) |
| `gt` | (N, S, S): uma máscara por verme do gabarito, sobreposição preservada (alvo de treino) |
| `meta` | quadro, bbox, `n_est`, `fits` |
