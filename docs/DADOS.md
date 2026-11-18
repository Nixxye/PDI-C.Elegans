# Dados

Os dados ficam em `data/` (ignorada pelo git). O conjunto abaixo é **CC BY 4.0**: citar ao usar.

| Dataset | Origem | O que é | Como entra no projeto |
|---|---|---|---|
| **CSB-1** (WormSwin) | [Zenodo 7456803](https://doi.org/10.5281/zenodo.7456803) — Deserno & Bozek, 2023; artigo: *WormSwin: instance segmentation of C. elegans using vision transformer*, Sci. Rep. Vídeos fornecidos por M. Rieckher; máscaras feitas sob supervisão de Bozek e Deserno | **Vídeo** de placa (912×736, cinza gravado como RGB), uma máscara por verme, com sobreposição | Direto: `--dataset-root data/csb-1_extracted/csb-1_dataset --seq <seq>`. É o conjunto usado nos resultados |

O `csb-1_dataset.zip` tem 10 sequências (`24_0_0_1` … `24_2_1_2`), `images/<seq>/<k>.png` e `masks/<seq>/<k>_<i>.png` (uma máscara por verme). O zip também traz `coco_annotations/`, que não é usado. A sequência `24_2_1_1` é a de **teste**; nenhuma ferramenta a usa para treinar.

**Atenção: o gabarito do CSB-1 tem máscaras duplicadas** (duas máscaras quase idênticas para o mesmo verme, IoU > 0,5; inspecionei 4 casos: é um verme só com dois contornos). Contagem de máscaras por sequência, 1 quadro a cada 10 (cru → após fundir as duplicadas, `tools/pdi_report.py`): `24_0_0_1` 287 → 284; `24_0_0_3` 422 → 415; `24_0_1_1` 250 → 250; `24_0_1_2` 273 → 273; `24_1_0_1` 581 → 564; `24_1_0_2` 371 → 371; **`24_1_1_1` 774 → 422**; **`24_1_1_2` 1.296 → 760**; `24_2_1_1` (teste) 942 → 941; `24_2_1_2` 922 → 868. Isso **rebaixa o recall** das métricas do estágio `evaluate` (cada duplicata conta como verme não encontrado) nas sequências `24_1_1_x`. A sequência de teste quase não tem duplicatas. O `evaluate` e o `export` **não** fundem as duplicadas; só o `pdi_report.py` (função `dedup`). Quem treinar a CNN deve fundi-las.

Layout esperado (ajuste `input.pattern`/`input.mask_pattern` se os nomes forem outros):

```
data/csb-1_extracted/csb-1_dataset/
├── images/<seq>/<k>.png
└── masks/<seq>/<k>_<i>.png
```

## Recortes para a CNN (`stages/export.py`)

Cada amontoado vira um recorte `patch_size × patch_size` (padrão 192) centrado nele, gravado em `<saída>/cnn_dataset/<seq>/f<quadro>_c<componente>.npz`:

| Campo | Conteúdo |
|---|---|
| `image`, `signal` | quadro em cinza e sinal pós-fundo (uint8) |
| `mask` | máscara binária do amontoado (só aquele componente) |
| `dist`, `skeleton` | transformada de distância e esqueleto (não são necessários para a rede) |
| `gt` | (N, S, S): uma máscara por verme do gabarito, sobreposição preservada (alvo de treino) |
| `meta` | quadro, bbox, `n_est`, `fits` |
