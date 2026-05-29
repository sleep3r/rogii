Да, это именно тот контекст, который нужен для большого разворота. После этих материалов я бы переформулировал наш план ещё точнее:

```text
Не GeoPathNet как generic TCN.
А MTP / MDN-style GeoSteering Inversion Network:
    2D misfit heatmap + SDF priors
    -> CNN encoder
    -> K trajectory hypotheses + probabilities
    -> sequential multi-realization tracking
    -> bounded blend / candidate bank
```

И это уже не “микрошаг”. Это понятная архитектура, которая совпадает с геофизической постановкой задачи.

---

# Главный вывод из материалов

Все источники бьют в одну точку:

> Геостиринг — это не регрессия TVT.
> Это неоднозначная стратиграфическая инверсия, где нужно держать несколько правдоподобных интерпретаций.

Статья Alyaev & Elsheikh прямо формулирует проблему как multi-modal probabilistic inversion of geophysical logs: модель получает horizontal и offset well logs, а на выходе даёт несколько вероятных SVD/stratigraphic trajectories и их вероятности. Они подчёркивают, что single deterministic regression хуже, потому что inverse problem неоднозначен, а вероятные альтернативные интерпретации нужны для решений в реальном времени. 

Постер Ambrus/Alyaev добавляет практический geosteering вариант: MDN выдаёт несколько stratigraphic interpretations, затем применяется sequentially, вероятные реализации трекаются, совпадающие точки merge-ятся, низковероятные реализации отбрасываются. Это почти прямое описание того, что нам надо сделать для ROGII. 

Webinar-транскрипт тоже попадает туда же: assisted geosteering tool показывает на 2D display “all potential interpretations”, а не одну линию; человек может держать 3–4 интерпретации, а инструмент смотрит тысячи вариантов. 

---

# Что это меняет в нашем плане

Раньше мы хотели:

```text
base path
+ cost-volume
+ TCN offset decoder
```

Теперь я бы сделал чуть иначе:

```text
MTP-CNN local inversion model
+ sequential hypothesis tracker
+ optional global bounded correction
```

То есть не просто “предсказать offset distribution per row”, а **предсказать K возможных путей на следующий chunk**, как в MTP notebook/paper.

---

# Архитектура: ROGII-MTPNet

## Название

```text
ROGII-MTPNet
```

или:

```text
GeoMTP
```

Формулировка:

```text
Input:
    локальный 2D heatmap matching horizontal GR ↔ typewell GR
    + history path mask
    + base/B2/A/SDF priors

Output:
    K возможных TVT/SVD trajectories на следующие L compressed steps
    + probability/logit каждого trajectory
```

Это ближе к notebook hengck23, где есть:

```text
[2D Heatmap Input] -> [Regression Head CNN] -> [MDN Predictor MLP] -> [Multi-Trajectory Output]
```

и где участник прямо говорит, что MDN predictor нужен для multiple path hypotheses “like k-beam”. 

---

# 1. Input representation

## 1.1. Compression

Из demo notebook:

```text
S = 32
heatmap shape = [64, 24]
history shape = [64, 24]
K = 10 paths
L = 24 path length
```

Это очень важный практический выбор: не пытаться кормить модель 7000 строками well целиком. Сначала работаем в compressed coordinates.

Для ROGII v0:

```text
S = 16 или 32 ft
T_typewell = 96 или 128 vertical bins
T_horizontal = 32 compressed MD steps
history_steps = 8–12
future_steps = 16–24
K = 8 или 10 modes
```

Я бы начал с:

```text
S = 32
H = 64 vertical bins
W = 24 horizontal steps
history = 8
future = 16
K = 8
```

Почему не сразу больше: сначала нужно проверить, что модель учится локальному inversion. Потом расширять horizon.

---

## 1.2. Heatmap channels

Базовый heatmap из статьи и notebook — это pairwise difference:

```text
heatmap[j, i] = horizontal_GR[i] - typewell_GR[j]
```

В статье прямо описано, что перед DNN input vectors pair-wise subtract друг из друга, формируя heat-map image, где пиксели — разность well-log и offset-log; output — несколько paths + logits/probabilities. 

Но для нашего v0 не надо ограничиваться одним каналом. Делай 8–12 каналов:

```text
ch0: GR_diff = h_gr - tw_gr
ch1: abs_GR_diff
ch2: dGR_diff
ch3: NCC_local_score
ch4: horizontal_GR broadcast
ch5: typewell_GR broadcast
ch6: GR_mask / finite mask
ch7: history_path_mask
ch8: base_path_SDF
ch9: B2_path_SDF
ch10: A_samples_density / A_SDF
ch11: formation_prior_SDF
```

Ключевое новое слово — **SDF**.

В обсуждении hengck23 прямо пишет, что одна из сложностей — хорошее representation, и предлагает CNN + SDF / signed distance function; дальше он говорит, что можно предсказывать geology plane, например `ANCC = tvt - z`, потому что такие planes более линейны и выигрывают от SDF/natural smoothness. 

Для нас SDF — это способ не давать CNN только noisy GR heatmap, а добавить геологические priors:

```text
SDF_to_base_path[j, i] = signed_distance(vertical_bin_j, base_tvt_at_i)
SDF_to_B2_path[j, i]
SDF_to_A_median_path[j, i]
SDF_to_A_p10_p90_band[j, i]
SDF_to_known_history_path[j, i]
```

---

## 1.3. History path

В notebook history рисуется как path-mask на heatmap: известные matched points до prediction start задаются отдельным каналом. Это очень важно: без history модель будет делать локальный GR matching, но не будет понимать, откуда пришёл well path.

Для ROGII:

```text
history_path = TVT_input known tail compressed into vertical bins
```

Если window начинается внутри hidden после sequential rollout:

```text
history_path = previous predicted top-K paths / selected path
```

---

# 2. Output

Модель не должна выдавать один TVT.

Выход:

```text
paths:  [B, K, L]
logits: [B, K]
```

Где:

```text
K = number of hypotheses
L = future compressed steps
paths[b,k,t] = predicted vertical-bin index or offset-bin
logits[b,k] = unscaled probability of mode k
```

Статья описывает ровно такой формат: DNN predicts predefined number of SVD function realizations and probability/logit for each path; output size is `M * (l+ + 1)`, то есть path points плюс probability для каждого mode. 

---

# 3. Target

Есть два варианта.

## Вариант A — vertical bin target

Как в notebook:

```text
target[t] = matched vertical bin index of true TVT in typewell grid
```

То есть модель учит path in heatmap coordinates.

Плюсы:

```text
ближе к MTP notebook
стабильнее
легче визуализировать
```

Минусы:

```text
нужно потом переводить bins -> TVT
```

## Вариант B — offset target относительно base

```text
target[t] = true_TVT[t] - base_TVT[t]
```

Плюсы:

```text
прямо оптимизирует correction к schema10/B2
легче bounded guard
```

Минусы:

```text
хуже совпадает с heatmap geometry
```

Я бы сделал **оба head-а**, но v0 начать с vertical-bin path.

Практически:

```text
primary path target = vertical bin index
aux target = TVT offset from base
```

---

# 4. Loss: MTP, а не обычный MSE

Главное отличие от TCN:

```text
не усреднять все возможные решения
а разрешить K разных trajectories
```

MTP loss:

```text
1. Для каждого sample считаем error каждого mode.
2. Выбираем closest mode m*.
3. Regression loss двигает только closest mode к truth.
4. Classification loss повышает probability closest mode.
```

В статье MTP loss состоит из classification loss для вероятности mode и best-mode regression loss; closest mode определяется как `argmin` расстояния до true trajectory, а regression loss уменьшает расстояние только для этого mode. 

В poster это то же самое: classification loss повышает вероятность mode, ближайшего к actual data, а regression loss тянет ближайшую prediction к actual data через MAE. 

Для нас:

```python
error[k] = mean(abs(pred_path[k] - target_path))
best_k = argmin(error)
loss = MAE(pred_path[best_k], target_path)
     + alpha_cls * CE(logits, best_k)
     + lambda_smooth * second_diff_penalty(pred_path[best_k])
```

Стартовые параметры:

```text
alpha_cls = 0.1 или 0.3
lambda_smooth = 0.01
K = 8
```

Важно: в статье сказано, что `alpha_class` регулирует ширину modes; слишком высокий может привести к mode collapse, слишком низкий требует больше modes. Это надо будет sweep-нуть маленьким grid. 

---

# 5. Backbone

Из paper architecture:

```text
Regression Head:
    2D conv + pooling layers
MDN:
    fully connected layers
Output:
    M paths + probabilities
```

Статья описывает RH как три пары 2D convolution `3x3` + max pooling `2x2`, затем fully connected layers, а MDN как dense predictor с широким FC layer и linear output. 

Для ROGII:

```text
CNN encoder:
    Conv2D 16
    Conv2D 16
    Pool
    Conv2D 32
    Conv2D 32
    Pool
    Conv2D 64
    Conv2D 64
    Pool
    Conv2D 128

Head:
    flatten
    FC 1024
    FC 2048
    path_head -> K * L
    logit_head -> K
```

Но я бы добавил **context vector**:

```text
context:
    hidden_len
    md_from_anchor
    last_tvt
    tail_slope
    GR_nan_ratio
    A/B2 danger features
    well-level geometry stats
```

В paper тоже отмечается, что extra contextual information can be passed to predictor. 

---

# 6. Sequential inference

Очень важное: не пытаться предсказать весь hidden interval одним window.

Делаем sequentially:

```text
start from known TVT_input tail
for step in chunks:
    build heatmap around current state
    model predicts K future path continuations
    combine with previous realizations
    keep top-N realizations
```

Постер описывает ровно это: at each interpretation step есть likely SVD functions from previous step and probabilities; final points previous interpretations become starting points for new interpretations; coinciding points are merged with increased probability, low-probability realizations are discarded. 

Для нас:

```text
N_realizations = 32 или 64
K_model = 8
keep_top = 32
merge tolerance = 2–4 ft
chunk_len = 16 compressed steps
stride = 8 или 16
```

Итоговые outputs:

```text
mtp_best_path
mtp_prob_weighted_mean
mtp_top3_mean
mtp_p10 / mtp_p50 / mtp_p90
mtp_entropy
mtp_mode_gap
mtp_num_active_modes
```

---

# 7. Почему это лучше нашего текущего B2

B2 уже доказал:

```text
A/B signals useful as bounded correction.
```

Но B2 не умеет:

```text
держать несколько interpretations
выбирать mode sequentially
собирать локальные GR micro-patterns
использовать heatmap как image
понимать ambiguity
```

MTPNet умеет именно это.

В обсуждении hengck23 пишет, что CNN может ловить micro 2D patterns, а GR-pair heatmap выглядит как 2D tokens; он также отмечает, что на longer horizon prediction diverges, но truth всё ещё часто есть как lower-score candidate/top-6, то есть нужна multi-mode tracking, а не один mean path. 

Это идеально совпадает с нашими A/B результатами:

```text
A oracle хорош, selector плох.
B true path видит, hard selector плох.
=> надо не hard select, а top-K probabilistic trajectory model.
```

---

# 8. Что именно делать в репе

## Новый пакет

```text
rogii/mtp/
    heatmap.py
    sdf.py
    dataset.py
    model.py
    loss.py
    train.py
    infer.py
    track.py
    evaluate.py
```

---

## Команды

```bash
make mtp-build-data \
  MTP_CONFIG=configs/mtp_v0.yml \
  MTP_OUTPUT=artifacts/mtp_v0_data

make mtp-train \
  MTP_CONFIG=configs/mtp_v0.yml \
  MTP_DATA=artifacts/mtp_v0_data \
  MTP_OUTPUT=artifacts/mtp_v0_model

make mtp-infer-oof \
  MTP_CONFIG=configs/mtp_v0.yml \
  MTP_MODEL=artifacts/mtp_v0_model \
  MTP_OUTPUT=artifacts/mtp_v0_oof

make mtp-eval \
  MTP_PRED=artifacts/mtp_v0_oof/predictions.parquet
```

---

# 9. MTP v0 config

```yaml
compression:
  rows_per_step: 32
  history_steps: 8
  future_steps: 16
  vertical_bins: 64
  vertical_radius_ft: 160

modes:
  K: 8

input_channels:
  gr_diff: true
  abs_gr_diff: true
  dgr_diff: true
  ncc_score: true
  gr_mask: true
  history_sdf: true
  base_sdf: true
  b2_sdf: true
  a_density: true
  a_sdf: true
  formation_sdf: false  # включить позже

base:
  path: c11_schema10_oof_pp
  b2_path: b2_guarded_submit

model:
  cnn_channels: [16, 32, 64, 128]
  fc: [1024, 2048]
  dropout: 0.05

loss:
  alpha_cls: 0.2
  smooth_lambda: 0.01
  path_loss: mae

tracking:
  keep_realizations: 32
  merge_tolerance_ft: 3
  probability_temperature: 1.0

validation:
  folds: 5
  group_by: well_id
```

---

# 10. Как валидировать

Нужны 4 уровня.

## Level 1 — local window accuracy

На windows:

```text
best-mode MAE/RMSE
prob-weighted mean RMSE
top1 mode RMSE
top3 oracle RMSE
top6 oracle RMSE
mode entropy
```

Успех v0:

```text
top3 oracle clearly better than weighted mean
truth often inside top-K
```

Это важнее, чем сразу финальный well RMSE.

---

## Level 2 — sequential OOF well path

После tracking:

```text
mtp_best_path RMSE
mtp_mean_path RMSE
mtp_top3_mean RMSE
mtp_p50 RMSE
P90/P95/worst well
long/short hidden
high GR NaN
```

Сравнить с:

```text
c11 base: 10.903
B2 guarded: 10.422
schema10 clean: 10.689 / public 10.084
```

Успех v0:

```text
MTP mean/top path <= 10.7
top-K oracle <= 9.5
```

Успех v1:

```text
MTP guarded blend <= 10.3
```

Цель v2:

```text
< 10.0 OOF or public sub-10
```

---

## Level 3 — blend with B2/base

MTP не обязан сразу быть standalone лучше B2. Проверить:

```text
final = base + alpha * clip(mtp_path - base)
final = B2 + alpha * clip(mtp_path - B2)
final = weighted blend(base, B2, mtp_mean, mtp_best)
```

Grid маленький:

```text
alpha: 0.1, 0.2, 0.3, 0.5
clip: 10, 20, 30
```

---

## Level 4 — danger guard

Все path models должны проходить те же guards:

```text
P95 shift
endpoint shift
one-sided shift
roughness ratio
worst wells
cross-fold positive
```

---

# 11. Go / No-Go

## Build is successful if

```text
window top6 oracle good
sequential top-K oracle <= 9.5
MTP guarded blend improves B2 by >= 0.15
P95/worst not worse
```

## Strong GO

```text
MTP/B2 blend <= 10.2
or standalone/mean <= 10.3
or public LB improves B2/schema10 clearly
```

## No-Go after two iterations

```text
top-K oracle > 10.0
truth not in top-K often
sequential tracking drifts catastrophically
MTP only learns base/B2 copy
```

Если No-Go, тогда MTPNet всё равно полезен как feature/candidate, но не финальная architecture.

---

# 12. Что делать прямо сейчас

Я бы теперь параллелил два трека.

## Track A — ship current B2

Это практический leaderboard signal:

```text
B2 inference parity
prediction guard
submit b2_guarded_submit
```

Не надо ждать MTP.

## Track B — build MTPNet v0

Это основной gold-track:

```text
1. Build heatmap/SDF data.
2. Train local window MTP CNN.
3. Evaluate top-K window oracle.
4. Add sequential tracker.
5. Blend/guard with B2.
```

---

# 13. Чего НЕ делать

```text
не делать ещё Surface Student v2
не тюнить A2/B2 selectors дальше
не делать generic Transformer на rows
не делать free HMM/DP на весь TVT без base bound
не делать full-well model первым шагом
```

Почему: paper и discussion говорят не “большой transformer решит всё”, а **multi-modal path hypotheses over heatmap**. Это разные вещи.

---

# 14. Финальный вывод

Ты принёс материалы, которые прямо подтверждают правильную большую архитектуру:

```text
1. Геостиринг неоднозначен → нужно несколько modes.
2. Правильный input — 2D log-mismatch heatmap, а не только tabular row features.
3. Правильный output — K trajectories + probabilities.
4. Правильный loss — MTP: best-mode regression + mode classification.
5. Правильный inference — sequential tracking/merging realizations.
6. SDF priors дают способ впихнуть base/B2/A/formation knowledge в CNN.
```

Это не отменяет B2. B2 — наш первый bounded submit candidate.
Но gold-track теперь такой:

```text
B2 guarded submit
→ MTP heatmap/SDF dataset
→ CNN-MTP local inversion
→ sequential multi-realization tracking
→ guarded blend with B2/base
→ candidate bank / public blend
```

И да: это уже не возня над копейками. Это полноценная постановка задачи в стиле материалов, которые ты принёс.
