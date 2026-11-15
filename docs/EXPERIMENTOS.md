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

### 1.2 Fechamento em tons de cinza do sinal (descartado, `background.close_kind`, desligado)

A ideia "pixels vizinhos com a mesma cor" (fechamento do sinal, também com linhas orientadas e suavização gaussiana). Resolve em sequências limpas (no teste, vermes partidos 8 → 1 de 760 com disco de raio 3), mas em `24_1_0_2` funde a sujeira em blocos (amontoados por quadro 0,5 → 7-8, F1 dos isolados cai a zero). Média das 5 sequências de treino com disco de raio 3: F1 dos isolados 0,71 → 0,57. Suavização gaussiana não ajudou (Dice 0,85 → 0,81).

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

### 1.6 Detecção de amontoado

Critério: área do componente ≥ `analysis.cluster_ratio` (1,5) × a área de referência (mediana das áreas dos componentes nos quadros de amostra; fixável com `analysis.ref_area`). Limitações: a sobreposição **reduz** a área visível (dois vermes cruzados somam menos que 2); dois vermes finos lado a lado podem ficar abaixo do limiar; um verme grande ou enrolado pode passar do limiar sem ser amontoado.

Métricas que não exigem separar o amontoado: `f1_single` (vermes isolados previstos × vermes do gabarito fora de amontoados) e `cluster_dice` (máscara de amontoados prevista × a do gabarito, onde amontoado = componente conexo do gabarito com ≥ 2 vermes). Com o padrão atual, `24_2_1_1`: `f1_single` ≈ 0,85 e `cluster_dice` ≈ 0,67.

O watershed por distância (Aula 11) é a alternativa clássica para separar o amontoado, mas fica **desligado** (`split.enabled: false`): vermes cruzados em X ou colados em paralelo não têm "pescoço" na máscara e o watershed erra. Nos recortes de amontoado do teste ele tem F1 0,69.

### 1.7 Sequência sintética (`tools/make_synthetic.py`)

8 vermes em 320×320, 60 quadros: Dice ≈ 0,88, F1 por instância ≈ 0,63-0,70 e só ≈ 30 % dos vermes dentro de amontoados recuperados (sem CNN).

## 2. CNN

### 2.1 Histórico (F1 por verme nos 574 amontoados do teste)

| Versão | F1 | Problema / achado |
|---|---|---|
| U-Net de 3 níveis (campo de visão ~60 px) | 0,75 | devolvia **sempre 2** vermes |
| 4 níveis + contexto global + amostragem balanceada | 0,82 | 2 ou 3; nunca 4+ |
| + cabeça de existência por canal, amostragem balanceada | 0,66 | inventava um 3º verme (155 de 427 amontoados de 2); variar o limiar de existência não resolveu (máx. 0,71) |
| idem, perda de máscara mais forte, amostragem uniforme | 0,80 | melhorou o caso de 1 verme (9 de 38), piorou o de 3 (17 de 70) |
| **+ amontoados sintéticos (colagem)** | **0,93** | chega a prever 4 e 5 vermes |
| mesma receita, **sem `24_2_1_2` no treino** (gravação vizinha da de teste) | 0,94 | descarta a suspeita de vazamento entre as duas sequências |

Só ~13 % dos recortes reais têm 3+ vermes (74 % têm 2) e a rede aprendia a devolver sempre 2. O que mais ajudou foi fabricar, em metade das amostras de treino, amontoados maiores **colando 1 a 3 vermes de outros recortes** (com a máscara exata) sobre o recorte, encostados no amontoado.

Usar a estimativa de contagem pela área (`n_est`) para escolher quantos canais manter foi testado e não ajudou: acerta 90 % no teste, mas só 29 % em `24_1_1_2`.

### 2.2 Contagem de vermes no teste (linha = no gabarito; entre parênteses, o previsto)

**1** → 2 em 35 de 38 (ruim); **2** → 2 em 407 de 427; **3** → 3 em 64 de 70; **4** → 4 em 6 de 15 (3 em 2, 5 em 7); **5+** → 5 em 15 de 24 (4 em 9).

### 2.3 Resultados no pipeline completo (`24_2_1_1`, passo 2)

| | F1 | recall | precisão | recall em amontoados |
|---|---|---|---|---|
| sem CNN | 0,74 | 0,75 | 0,74 | 0,05 |
| com CNN | 0,90 | 0,97 | 0,85 | 0,91 |

`f1_in_cluster` (vermes separados dentro de amontoado × vermes do gabarito que estão em amontoado): 0 → 0,66.

### 2.4 Informação temporal (quadro anterior): testada e **removida do código**

Ideia: dar à rede o que o sistema viu no quadro anterior. Duas formas foram implementadas e testadas; o código foi tirado para simplificar o projeto e está no histórico do git (commit `6d7095e`, "Resultados iniciais", com os pesos `models/cluster_cnn_temporal.pt`):

- **5 canais de entrada** (`train_cnn.py --temporal`): os 3 de sempre mais a união das máscaras dos vermes do quadro anterior perto do amontoado e a contagem deles ÷ 6.
- **Dica de contagem na saída** (`refine.use_prev_count`): mantém os *n* canais mais prováveis, *n* = vermes do quadro anterior.

| Recorte, **quadro anterior = gabarito** (limite superior) | F1 | Contagem certa |
|---|---|---|
| 3 canais, limiar de existência (padrão) | 0,928 | 0,84 |
| 3 canais + n do quadro anterior | 0,932 | 0,92 |
| 5 canais, limiar de existência | 0,931 | 0,83 |
| 5 canais + n do quadro anterior | 0,941 | 0,93 |

| Pipeline completo, **quadro anterior = saída do sistema** | F1 | recall | recall em amontoados | `f1_in_cluster` |
|---|---|---|---|---|
| **3 canais, sem dica (padrão)** | **0,902** | 0,971 | 0,907 | **0,658** |
| 5 canais | 0,892 | 0,958 | 0,857 | 0,622 |
| 5 canais + n do quadro anterior | 0,894 | 0,966 | 0,881 | 0,632 |
| 3 canais + n do quadro anterior | 0,893 | 0,973 | 0,912 | 0,624 |

O número de vermes do quadro anterior é muito informativo (bate com o real em 91 % dos recortes quando vem do gabarito), mas a rede de 5 canais quase não o usa, e mesmo no limite superior o ganho é pequeno (+0,013 de F1). No pipeline real o número vindo da saída do sistema bate com o real em só **78 %** (primeiros 120 quadros), principalmente por excesso (amontoados de 2 vermes com 3, 4 ou 6), e uma contagem a mais acrescenta um verme falso. Não foi implementada a versão de um canal por verme.

## 3. O que não foi feito ou investigado

- **Rastreamento.** Um rastreador simples (associação por algoritmo húngaro com predição de posição) foi implementado e **removido** por enquanto: gerava ≈ 60 IDs para ≈ 20 vermes (trilhas fragmentadas, trocas dentro de amontoados). Seguir cada verme no vídeo continua sendo o objetivo final.
- **`24_1_1_2`**: a CNN tem F1 0,81 (watershed 0,64), mas só 29 % de contagem certa (devolve 2 para quase todo amontoado de 3-4). Causa não investigada (talvez tamanho/aparência dos vermes ou o gabarito dessa sequência).
- Emaranhados de 4+ vermes, vermes paralelos colados e o verme enrolado sobre si ainda erram bastante (`python tools/cnn_check.py --worst` desenha os piores casos).
- Parâmetros em pixels (`min_area`, `marker_min_dist`, `tophat_size`) foram calibrados no CSB-1 (912×736); revise-os para outra resolução.
