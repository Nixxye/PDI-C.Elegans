# Experimentos e decisões

Registro do que foi testado, com os números, para explicar por que cada parâmetro é o que é (e para não repetir o que não funcionou). Todos os números são do CSB-1. "Treino" aqui são as sequências usadas para **escolher** parâmetros ou treinar; **teste** é `24_2_1_1`. Cada tabela diz quantas sequências entram; os valores vêm de rodar o pipeline com `--set` (ver `README.md`).

> Ressalva geral: o teste é **uma sequência só**, e `labeling.min_area=150` foi escolhido olhando essa mesma sequência numa varredura pequena. Diferenças de 0,01 a 0,03 podem ser ruído.

## 1. Segmentação (PDI)

### 1.1 Verme parado entra no fundo → verme partido ou com a cauda faltando

O fundo é estimado com vários quadros (`sinal = fundo − quadro`). Se uma parte do verme fica parada por muitos quadros, o fundo "aprende" essa parte e o sinal dela some. Dois sintomas: o verme sai **partido em dois**, ou **sem a cauda**. Medido com o gabarito, em vermes isolados:

| Percentil do fundo (50 = mediana) | vermes partidos (9 sequências sem a de teste) | Dice médio | F1 dos vermes isolados |
|---|---|---|---|
| 50 | 2,3 % | 0,75 | 0,63 |
| **70 (padrão)** | **0,6 %** | 0,80 | **0,69** |
| 75 | 1,0 % | 0,80 | 0,69 |
| 80 | 1,4 % | 0,80 | 0,67 |

Percentis ≥ 80 pioram a sequência `24_1_0_2` (ágar cheio de riscos escuros): surgem falsos positivos grandes (média de 2,9 amontoados por quadro, contra 0,17 com a mediana).

### 1.2 Fechamento em tons de cinza do sinal (descartado e removido do código)

Fechar o sinal (disco, linhas orientadas ou suavização gaussiana) resolve vermes partidos em sequências limpas, mas em `24_1_0_2` funde a sujeira em blocos (F1 dos isolados cai a zero; média das 5 sequências de treino 0,71 → 0,57). Não foi mantido.

### 1.3 Ponte entre pedaços (ligado: `labeling.bridge_gap: 8`)

Liga com um traço reto os pares de componentes **grandes** (≥ `bridge_min_area` = 150 px) separados por até 8 px. Vermes partidos 0,6 % → 0,1 % (teste: 8 → 2 de 760). Custo: dois vermes distintos muito próximos também são ligados e viram amontoado (teste: F1 0,77 → 0,73; `f1_single` 0,85 → 0,83). Ligar também pedaços pequenos (60 px) gerou amontoados falsos. Gap 12 e 16 cortam mais, mas o F1 dos isolados cai para 0,81 e 0,77. Desligue com `--set labeling.bridge_gap=0`.

### 1.4 Corpo ou cauda incompletos (ligado: `threshold.grow_dist: 15`)

O defeito mais frequente: no teste, **22 %** dos vermes isolados ficavam com menos de 80 % do gabarito coberto (14 % no treino). `grow_dist` usa a máscara temporal como semente (≥ 150 px) e a completa com o sinal espacial (top-hat) até 15 px dela, com o mesmo limiar do sinal temporal. Resultado: cobertura < 80 % cai para ≈ 5 %. Custo: nas sequências com muita sujeira (`24_1_0_2`, `24_0_0_3`) o Dice cai (0,76 → 0,65 e 0,84 → 0,77) porque a sujeira junto do verme entra na máscara, e o pipeline fica ≈ 60 % mais lento. Top-hat sozinho no lugar do fundo temporal é pior (Dice 0,49 no treino).

### 1.5 Espinhos na máscara (ligado: `threshold.grow_open: 3`)

O top-hat também marca textura do ágar encostada no verme, que entra como espinho (aparece em laranja no painel 5 do `step_viewer`). Uma abertura de raio 3 só no trecho acrescentado tira os espinhos finos e mantém a cauda:

| `grow_open` | cobertura < 80 % (treino) | Dice (treino) | `f1_single` (treino) | Dice (teste) | `f1_single` (teste) |
|---|---|---|---|---|---|
| sem crescimento | 13,9 % | 0,803 | 0,698 | 0,846 | 0,837 |
| 0 | 4,7 % | 0,777 | 0,690 | 0,841 | 0,842 |
| 1 | 4,7 % | 0,778 | 0,691 | 0,842 | 0,842 |
| 2 | 4,8 % | 0,789 | 0,701 | 0,849 | 0,846 |
| **3 (padrão)** | 5,9 % | **0,801** | **0,713** | **0,852** | **0,851** |

### 1.6 Vermes não marcados (ligado: `threshold.spatial_min_area: 300`)

Medido com o gabarito em 10 sequências (6.118 vermes): **337 (5,5 %) não eram marcados** (cobertura final < 20 %). Todos se perdiam **no limiar** (nenhum na morfologia nem no `min_area`; **nenhum encosta na borda**), com sinal temporal quase zero (0,07 contra 0,24 nos marcados), concentrados em `24_1_1_2` (154), `24_1_1_1` (114) e `24_2_1_2` (62): são vermes parados que o fundo temporal absorveu. Como a perda é anterior à morfologia, **a erosão (ou a abertura) da Aula 08 não traria esses vermes de volta; ela só encolhe a máscara.** O que resolve é detectar pelo top-hat, que não depende de o verme se mover: componentes do top-hat (depois da abertura de raio `grow_open`) com ≥ 300 px entram na máscara mesmo sem semente temporal.

| `spatial_min_area` | vermes não marcados | precisão da máscara (pixels) | recall da máscara (pixels) | componentes por quadro |
|---|---|---|---|---|
| 0 (desligado) | 337 | 0,755 | 0,868 | 11,8 |
| 450 | 14 | 0,750 | 0,921 | 12,3 |
| **300 (padrão)** | **0** | 0,746 | 0,922 | 12,4 |

**Ressalva:** o recall dessas sequências ainda é baixo (≈ 0,55-0,58 a IoU 0,5) quando o gabarito tem duplicatas: os vermes parados agora são marcados, mas a máscara deles muitas vezes não chega a IoU 0,5 (cobertura parcial ou fundida com sujeira).

### 1.7 Detecção de amontoado

Critério: área do componente ≥ `analysis.cluster_ratio` (1,5) × a área de referência (mediana das áreas dos componentes nos quadros de amostra; fixável com `analysis.ref_area`). Limitações: a sobreposição **reduz** a área visível (dois vermes cruzados somam menos que 2); dois vermes finos lado a lado podem ficar abaixo do limiar; um verme grande ou enrolado pode passar do limiar sem ser amontoado.

Métricas que não exigem separar o amontoado: `f1_single` (vermes isolados previstos × vermes do gabarito fora de amontoados) e `cluster_dice` (máscara de amontoados prevista × a do gabarito, onde amontoado = componente conexo do gabarito com ≥ 2 vermes). Com o padrão atual, `24_2_1_1`: `f1_single` ≈ 0,85 e `cluster_dice` ≈ 0,67.

O watershed por distância (Aula 11) foi a alternativa clássica testada para separar o amontoado e foi **removido do código** (está no git, commit `3ecc77f`): vermes cruzados em X ou colados em paralelo não têm "pescoço" na máscara e ele erra. Nos recortes de amontoado do teste teve F1 0,69.

### 1.8 Sequência sintética (`tools/make_synthetic.py`)

8 vermes em 320×320, 60 quadros: Dice ≈ 0,88, F1 por instância ≈ 0,63-0,70 e só ≈ 30 % dos vermes dentro de amontoados recuperados (sem CNN).

### 1.9 Falsos positivos e filtro de forma (ligado: `analysis.min_mean_width: 5`, `analysis.max_mean_width: 40`)

`tools/pdi_report.py` conta como falso positivo um componente com menos de 30 % da área sobre o gabarito, em **dois tipos**: **tipo 1, isolado** (o PDI não o marcou como amontoado) e **tipo 2, amontoado** (vai para a CNN, mas quase não tem verme). Dois tipos de componente sem gabarito **não** contam: (1) **os que encostam na borda do quadro**, porque o gabarito do CSB-1 não anota os vermes cortados pela borda (**737 dos 924 "falsos positivos" iniciais, 80 %**); (2) **os que têm forma e contraste de verme** (`evaluate.classify_no_gt`: área 0,7 a 1,4 × a de um verme, largura média 6,5 a 10,5 px, sinal espacial médio ≥ 0,2), que são vermes reais que o gabarito simplesmente não anotou (**28**; ex.: `24_1_0_2` quadro 310). Essa regra classifica 95,8 % dos isolados que **têm** verme no gabarito como "forma de verme", e dos 28 componentes assim, **12 de 12 amostrados eram vermes reais**; os 159 sem forma de verme que sobram são rabiscos e sujeira do ágar (também conferido numa amostra de 12). Resultado: **159 falsos positivos (2,8 % dos 5.637 componentes; tipo 1: 155, 3,0 % dos isolados; tipo 2: 4, 0,9 % dos amontoados)**, mais 737 "sem gabarito na borda" e 28 "sem gabarito, com forma de verme" em linhas à parte. Dos 159, **140 estão em `24_1_0_2`** (a sequência com mais sujeira no ágar). Numa amostra de 12 dos componentes sem gabarito que **não** são da borda e sem a regra de forma de verme, 8 eram vermes reais não anotados, 2 manchas escuras redondas e 2 emaranhados de sujeira; foi isso que levou à regra de forma de verme acima.

Medi 4.722 componentes (7 sequências; "com verme" = ≥ 60 % sobre o gabarito: 3.678; "sem verme": 966) para achar um critério de forma que tire manchas e sujeira **sem perder verme**. Critério: largura média = área ÷ comprimento do esqueleto (verme: mediana 8,2 px).

| Regra | com verme perdidos | sem verme removidos |
|---|---|---|
| largura média < 5 px (fino demais: sujeira) | 0 | 138 (14,3 %) |
| largura média ≥ 14 px (mancha) | 21 (0,57 %) | 54 |
| largura média ≥ 30 px | 1 | 41 |
| largura média ≥ 40 px | **0** | 41 |
| **< 5 px ou ≥ 40 px (padrão)** | **0** | **179 (18,5 %)** |
| espessura ≥ 22 px | 35 | 41 |
| junções do esqueleto por 100 px | não separa (verme também tem) | |

Vermes enrolados em bola têm largura média alta (até ~33 px; 16 a 24 px é comum), por isso o limite de cima tem de ser 40, e não os 14 a 20 px que "parecem" manchas. A intensidade do componente também não separa as manchas.

No relatório completo (10 sequências), sem × com o filtro: vermes encontrados 100 % × 100 %; acerto da máscara dos isolados 99,9 % × 99,9 %; IoU médio 0,797 × 0,797; **falsos positivos (os dois tipos) 1.103 × 924 antes da regra da borda e da regra de forma de verme; 159 com elas** (o filtro de forma tirou 179 componentes sem verme: tipo 1 de 1.091 para 913 e tipo 2 de 12 para 11, antes de descontar borda e verme não anotado), sobretudo em `24_1_0_2` (326 → 200 antes das regras de borda/verme) e `24_2_1_2` (184 → 138). Restam 2,8 % dos componentes como falso positivo. Dois tipos que o filtro **não** pega: **verme real não anotado** (não é erro do PDI) e **verme com sujeira colada** (o crescimento pelo top-hat, `grow_dist`, puxa a textura em volta e o componente fica com < 30 % de área sobre o verme e geralmente é marcado como amontoado: conta como falso positivo **tipo 2**, mas a CNN ainda o recebe). Desligue o filtro com `--set analysis.min_mean_width=0 --set analysis.max_mean_width=0`.

**"Verme pequeno" (decisão, não medição).** Os 159 falsos positivos restantes são pequenos (área mediana 255 px, comprimento do esqueleto 37 px, contraste 0,14), e os vermes do **gabarito são todos grandes** (área p10-p90 = 514-673 px, esqueleto 81 px, contraste 0,22-0,31; mesmo em `24_1_0_2`). Por isso **não dá para derivar do gabarito** o que é um verme pequeno: só 1 dos 159 cai nas faixas de área e largura dos vermes anotados. Seguindo a sua avaliação (viu a galeria e a maioria parece verme pequeno), um componente sem gabarito, fora da borda e sem forma de verme normal, com **área ≥ 150 px e contraste ≥ 0,1**, passa a ser "provável verme pequeno" e **não** conta como falso positivo (`evaluate.SMALL_WORM_MIN_AREA`, `SMALL_WORM_MIN_CONTRAST`; no relatório, `--small-min-area` e `--small-min-contrast`). Com esses valores **todos os 159 viram "verme pequeno" e os falsos positivos caem a 0**: isso vale **por definição**, sem prova de que sejam vermes (alguns podem ser sujeira escura). Como 150 px é o `labeling.min_area`, o limite de área sozinho não filtra nada. Se apertar o limite de área, os falsos positivos reaparecem: pela distribuição das áreas, com 300 px voltam ≈ 98; com 450 px, ≈ 138. A galeria (`--galeria-categorias pequeno`) mostra um por um os que foram reclassificados.

## 2. CNN (removida; a cargo do colega)

O código da CNN (U-Net com 6 canais de saída, existência por canal, treino com amontoados sintéticos, experimento com o quadro anterior) foi tirado do repositório. O que ficou é o esqueleto `celegans_seg/cnn.py` (ver README). O histórico completo está no git (commits `6d7095e` e `3ecc77f`). Resultados daquela versão, só como referência: F1 por verme nos 574 amontoados do teste 0,93 (watershed 0,69); o que mais ajudou foi colar vermes de outros recortes para fabricar amontoados grandes no treino; informação do quadro anterior deu ganho desprezível (+0,013 de F1 no limite superior).

## 3. Linha do tempo e filtro de Kalman (`stages/timeline.py`)

> O código do Kalman (`celegans_seg/kalman.py`, opção `timeline.kalman`) **não está neste commit**: foi revertido a pedido e ficou guardado em `git stash` ("backup antes de voltar ao ultimo commit"). Os resultados abaixo foram mantidos.

Liga as máscaras de quadros vizinhos pelo SSIM [Aula 05, pág. 19-22] (forma e sobreposição), com atribuição húngara e tolerância de 2 quadros sem aparecer. Medida contra o gabarito nos 120 primeiros quadros da 24_2_1_1 (`tools/timeline_eval.py`; trocas de ID = vezes em que o mesmo verme do gabarito muda de ID):

| versão | F1@0,5 | trocas de ID | fragmentações |
|---|---|---|---|
| CNN no quadro inteiro + rastreador por IoU | 0,972 | 5 | 17 |
| PDI + linha do tempo SSIM | 0,772 | 6 | 10 |
| PDI + CNN + linha do tempo SSIM | 0,921 | **5** | **10** |
| PDI + CNN + linha do tempo SSIM + Kalman | 0,921 | 22 | 10 |

A linha "PDI + CNN" melhorou em relação a uma medida anterior (21 trocas, F1 0,906) porque a CNN passou a separar também os amontoados grandes (janela maior). Só com o SSIM de forma (cada máscara recortada no próprio centroide) eram 119 trocas; somar o SSIM de posição levou a 21 e depois a 5.

**Filtro de Kalman** (`celegans_seg/kalman.py`, `timeline.kalman`, desligado por padrão). Velocidade constante em (y, x); a previsão substitui a última posição no limite de distância e no custo, e desloca a máscara anterior antes do SSIM de posição; limite de Mahalanobis. Escolha dos parâmetros na sequência de validação `24_0_0_1` (grade q ∈ {0,5; 1; 2}, r ∈ {2; 4}): ficou em q = 1, r = 4. **Resultado: não ajudou.**

| sequência | sem Kalman | com Kalman (q, r testados) |
|---|---|---|
| `24_0_0_1` (validação, 120 quadros) | 0 trocas | 4 a 7 |
| `24_2_1_1` (teste, 120 quadros) | 5 trocas | 18 a 28 |

Diagnóstico na sequência de teste: tirar o limite de Mahalanobis não muda (22 trocas), e quanto **mais** o filtro confia na velocidade (q menor), pior (q = 0,2 e r = 1: 25 trocas; q = 5 e r = 10: 18). Ou seja, o movimento do verme **não é bem previsto por velocidade constante** (ele se contorce e muda de direção), e a posição prevista atrapalha mais do que a última posição conhecida. Não testei outros modelos (aceleração, estado com área e orientação, filtro estendido) nem o Kalman só para suavizar o centroide. A sequência de validação quase não tem amontoados (0 trocas sem Kalman), então ela quase não testa o caso em que o Kalman deveria ajudar.

**De onde vêm as trocas de ID** (PDI + CNN, `24_2_1_1`, 120 quadros, 5 trocas; instâncias capturadas e o rastreador reexecutado offline). Em 4 das 5 o ID **não foi perdido na ligação**: a máscara do verme ficou ruim por 3 a 20 quadros (IoU < 0,5 com o gabarito) e, quando voltou ao normal, a trilha já tinha sido encerrada ou a nova máscara parecia diferente demais (SSIM abaixo de 0,35).

| causa | eventos | o que se vê |
|---|---|---|
| máscara parcial por 3-4 quadros (área 350-490 px, contra 640 no gabarito) | 2 | a trilha continua com o mesmo ID, mas a máscara completa que volta não casa com a parcial guardada |
| dois vermes fundidos em uma máscara só (área ≈ 1.700 px, ~2,5 vermes) por 8 quadros, não separada pela CNN | 2 | a máscara leva um só ID; o outro verme perde a trilha e ganha ID novo na separação |
| máscara errada por 20 quadros (outra máscara ocupa o lugar) | 1 | trilha encerrada pelo limite de falhas |

Testes offline (trocas nas sequências de teste / validação): parâmetros (`min_ssim`, `max_gap`, `max_dist`, `dist_weight`) variam entre 4 e 6 trocas na de teste, ou seja, **ruído**; `min_ssim` 0,5 ou `max_gap` 30 pioram (10 e 8). Uma segunda passada por distância para quem sobra deu 4 (uma troca a menos, dentro do ruído). Uma memória de fusão simples (manter vivas as trilhas cobertas por uma máscara grande) piorou muito (14 a 59), então não foi adotada. **Conclusão:** o limite do rastreador é a qualidade das máscaras; trocar o algoritmo de ligação muda pouco. São 5 eventos em uma sequência, então os números pequenos não distinguem variantes.

**Separação guiada pelas trilhas** (`timeline.split_by_tracks`, ligada; `split_ratio` 1,4, `split_cover` 0,3, `split_min_part` 40). Ataca a causa dos amontoados que a CNN deixou inteiros: quando uma máscara cobre 2 ou mais trilhas do quadro anterior e é ≥ 1,4 vez maior que a maior delas, cada pixel vai para a trilha de máscara anterior mais próxima (distância até a máscara, Aula 01c) e cada parte (≥ 40 px) vira um verme com o ID da sua trilha. PDI + CNN, 120 quadros:

| sequência | sem | com |
|---|---|---|
| `24_2_1_1` (teste) | F1 0,921 · FN 47 · 5 trocas · 10 fragm. | F1 0,923 · FN 34 · **3 trocas** · **7 fragm.** |
| `24_2_1_2` | F1 0,903 · FN 110 · 0 trocas · 32 fragm. | F1 0,909 · FN 77 · 0 trocas · 26 fragm. |
| `24_0_0_1`, `24_1_1_1` | sem amontoados fundidos | idêntico |

Razão 1,2 piora (F1 0,903 na 24_2_1_1); 1,4 a 1,7 dão resultados parecidos. A melhora é pequena e vem de poucos eventos, e nenhum parâmetro foi tirado de uma sequência separada do teste. A `24_1_1_1` aparecia com F1 0,686 e 966 vermes "não encontrados": **não era falha de detecção, e sim o gabarito**: 1.932 de 2.108 máscaras estão duplicadas (cada verme anotado duas vezes, IoU ≥ 0,5 entre elas no mesmo quadro). Fundindo as duplicadas (fica a maior) o gabarito cai a 1.142 vermes e o resultado vira F1 0,966, FN 0 (FP 80). O mesmo vale em menor grau para a `24_2_1_2` (76 duplicadas: F1 0,903 → 0,916, FN 110 → 34); `24_2_1_1` e `24_0_0_1` quase não têm. `tools/timeline_eval.py` agora funde as duplicadas por padrão (`--raw-gt` mantém cru; `mrcnn_common.dedup_gt`); `tools/pdi_report.py` já fazia isso. Os números das tabelas desta seção para a `24_2_1_1` não mudam; os da `24_2_1_2` acima estão com gabarito cru. Máscaras parciais (as outras 2 trocas) continuam sem solução.

**O que ainda erra na `24_2_1_1`** (PDI + CNN + separação guiada, 120 quadros, gabarito sem duplicatas: FP 363, FN 34).
- **FN (34):** nenhum verme sumiu por completo; todos são vermes achados com máscara que não bate (IoU < 0,5): 14 menores, 13 de forma diferente, 7 maiores. A maioria fica em um emaranhado de 3 vermes que se cruzam (quadros 63 a 87), onde o contorno do conjunto está certo e o que erra é o corte entre eles.
- **FP (363):** 282 são vermes **na borda da imagem**, que o gabarito não anota (a mesma ressalva da seção 1); só 81 estão fora da borda. Descontando os da borda o F1 vai de 0,923 a **0,976**.
- **Repartir cada emaranhado entre as trilhas do quadro anterior** (pixel → máscara anterior mais próxima), em vez de aceitar o corte da CNN: **piorou** (F1 0,920 e FN 48 quando o número de trilhas bate com o de máscaras; F1 0,838, FN 272 e 7 trocas quando basta cobrir 2 trilhas). O corte da CNN dentro do emaranhado é melhor que a posição do quadro anterior; não adotado.
- **Completar máscaras parciais com a imagem** (quando a área cai abaixo de 0,8 da referência da trilha, a máscara cresce a partir da anterior, só perto dela e onde o sinal ≥ fração do sinal médio da máscara; limite de 1,25 da referência), PDI + CNN, 120 quadros: raio 3 px quase não cresce (447 → 448 px, referência 624), sem efeito no resultado; raio 8 ou 15 com sinal 0,3: F1 0,925, FN 31, mas 4 trocas e 8 fragmentações (contra 0,923, 34, 3 e 7). Diferença dentro do ruído, **código removido**. Não investiguei por que a máscara fica parcial (hipóteses: verme parado absorvido pelo fundo, cauda com pouco sinal).
- **Usar o quadro seguinte (não causal)**, simulado offline sobre a saída do pipeline: por trilha, quadros cuja área foge ±25 % da mediana (janela de 15 quadros) e que ficam entre dois quadros bons de área parecida (falha de até L = 1, 3, 5 ou 10 quadros) têm a máscara refeita misturando a anterior e a seguinte, alinhadas pelo centroide interpolado. **Sem efeito**: 8 a 13 quadros ajustados na `24_2_1_1` e nenhum na `24_2_1_2`; F1 0,923, FN 34, 3 trocas e 7 fragmentações iguais ao causal. As falhas que sobram são longas (25 quadros seguidos no emaranhado) ou não mudam a área, então a interpolação entre vizinhos não as alcança. Não implementado no pipeline.
- **Memória de amontoado** (`timeline.hide_merged`, ligada; `hide_cover` 0,3, `hide_near` 30 px, razão de área 0,5 a 1,8, até 300 quadros): a trilha que fica sem máscara própria dentro da máscara de outra (verme que entrou em um amontoado) fica **oculta**, guardando a última máscara e a área de referência (que não cresce com a área do amontoado). Quando uma máscara nova surge a até 30 px do amontoado, com área entre 0,5 e 1,8 da referência, ela volta com o **mesmo ID** (atribuição húngara entre máscaras novas e trilhas ocultas; custo = diferença de área + distância). Resultado (120 quadros, PDI + CNN, gabarito sem duplicatas): `24_2_1_1` de 3 para **2 trocas** (some a do verme que passou 26 quadros dentro de um emaranhado); `24_2_1_2`, `24_0_0_1` e `24_1_1_1` sem mudança (0 trocas). As 2 trocas que sobram são de máscara parcial, não de amontoado. Limites: um evento corrigido; se uma máscara nova de área parecida aparecer perto do amontoado por outro motivo, ela pode herdar o ID errado (não observado).
- Conclusão da busca por melhorias temporais: depois da separação guiada e da memória de amontoado, mais nada ajudou. O que resta é o corte entre vermes dentro de emaranhados, que é a CNN (fora do que o PDI controla).

## 4. O que não foi feito ou investigado

- **Rastreamento.** Um rastreador simples (associação por algoritmo húngaro com predição de posição) foi implementado e **removido** por enquanto: gerava ≈ 60 IDs para ≈ 20 vermes (trilhas fragmentadas, trocas dentro de amontoados). Seguir cada verme no vídeo continua sendo o objetivo final.
- Parâmetros em pixels (`min_area`, `tophat_size`) foram calibrados no CSB-1 (912×736); revise-os para outra resolução.
