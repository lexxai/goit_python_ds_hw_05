# Домашнє завдання 05

В домашньому завданні до даного модулю ви потренуєтесь робити тестове завдання для влаштування на роботу. За даними акселерометра з мобільного телефону потрібно класифікувати, якою діяльністю займається людина: йде, стоїть, біжить чи йде по сходах. Знайти датасет ви можете за [посиланням](https://drive.google.com/file/d/1nzrtQpfaHL0OgJ_eXzA7VuEj7XotrSWO/view?usp=share_link).

Використайте алгоритми SVM та випадковий ліс з бібліотеки `scikit-learn`. Як характеристики можете брати показники з акселерометра, проте щоб покращити результати роботи алгоритмів, спочатку можна підготувати наш датасет і розрахувати часові ознаки (*time domain features*). Більше ці характеристики описані в даній [статті](https://drive.google.com/file/d/1-18YEmp0YjV3hN9iI8J1i_FWd55HFwOK/view?usp=share_link).

- Порівняйте результати роботи обох алгоритмів на різних фічах та різні моделі між собою.

- Порівняйте результати роботи обох алгоритмів на різних фічах та різні моделі між собою. Використайте метод `classification report` для порівняння.

- Порівняння моделей на основі однієї метрики(такої як `Accuracy`)- не приймається. 

Дз повинно бути виконано у Jupyter Nootebook,(або Google Colab) і задеплоїне на Гітхаб у вигляді файлу .ipynb.



# Виконнаня:

- [goit_ds_hw_05.ipynb](goit_ds_hw_05.ipynb)

- [Colab](https://colab.research.google.com/drive/1DD8nHq-iAc_k8v14iIDKtyw8NGoCkAyw?usp=sharing)

- [Colab Extra Tasks](https://colab.research.google.com/drive/1LOxtCv4zACK-iir-kigvzlt96_G-PBk3?usp=sharing)

# Результати:

Різні набори даних були створені з файлів у форматі CSV, завантажених з різних тек, кожен з яких був названий відповідно до відповідної діяльності. І кожен файл мав 30 записів з 3-х параметрів (координат) акселерометра.

* Набір даних f0s0: містить дані за 4 характеристиками.
* Набір даних f0s1: Містить дані з 31 характеристикою.
* Набір даних f1s0: містить дані з 91 ознакою.
* Набір даних f1s1: Містить дані зі 118 ознаками.

У наборі даних "f0s0" модель SVC-Linear була пропущена під час тривалої роботи, що потенційно перевищувала одну годину.

Моделі, використані для аналізу, включали SVC, SVC-Linear та RandomForestClassifier.

* Для набору даних з 4 ознаками (f0s0) модель RandomForestClassifier показала кращі результати.
* Для набору даних з 31 ознакою (f0s1) краще працює RandomForestClassifier.
* Для набору даних з 91 ознакою (f1s0) RandomForestClassifier працює краще.
* Для набору даних зі 118 ознаками (f1s1) RandomForestClassifier працює краще.

Цей набір даних 'f0s1' досягнув ідеальної точності 1.0000, що робить його найвищим серед наданих наборів даних.

Набір даних 'f0s1' не був вирівняний і до нього були додані статистичні характеристики.

Переможцем став набір даних 'f0s1' з 31 ознакою при використанні моделі RandomForestClassifier.


## Extra Tasks

Крім того, модель SVC було доповнено функцією StandardScaler(), що призвело до незначного збільшення точності на 0,452%.
```
Data set: f0s1, shape: (193860, 31), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000     30672
     walking     1.0000    1.0000    1.0000     30672
     running     1.0000    1.0000    1.0000     30672
      stairs     1.0000    1.0000    1.0000     30672

    accuracy                         1.0000    122688
   macro avg     1.0000    1.0000    1.0000    122688
weighted avg     1.0000    1.0000    1.0000    122688
--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000     30672
     walking     1.0000    1.0000    1.0000     30672
     running     1.0000    1.0000    1.0000     30672
      stairs     1.0000    1.0000    1.0000     30672

    accuracy                         1.0000    122688
   macro avg     1.0000    1.0000    1.0000    122688
weighted avg     1.0000    1.0000    1.0000    122688
```

- Для набору даних з 4 ознаками (f0s0) модель RandomForestClassifier показала кращі результати.
- Для набору даних з 31 ознакою (f0s1) краще працює як RandomForestClassifier так і SVC + StandardScaler().
- Для набору даних з 91 ознакою (f1s0) RandomForestClassifier працює краще.
- Для набору даних зі 118 ознаками (f1s1) RandomForestClassifier працює краще.
    
Цей набір даних 'f0s1' досягнув ідеальної точності 1.0000, що робить його найвищим серед наданих наборів даних як для RandomForestClassifier так і SVC з StandardScaler().

Час витарчений для навчання та прогнозування відповідно:
```
- classification: SVC
CPU times: user 1min 28s, sys: 222 ms, total: 1min 28s
Wall time: 1min 29s
CPU times: user 28.9 s, sys: 22.1 ms, total: 28.9 s
Wall time: 28.9 s
-----------------------------------
- classification: RandomForestClassifier
CPU times: user 38.8 s, sys: 39.9 ms, total: 38.8 s
Wall time: 38.9 s
CPU times: user 828 ms, sys: 2 ms, total: 830 ms
Wall time: 824 ms
```
Згідно цього вибору за рахункок швидкості виграє RandomForestClassifier.

### Балансування записів з додаванням синтеничних данних
#### SMOTE
Додано додковий reshahping з викорисанням [SMOTE](https://medium.com/thecyphy/handling-imbalanced-datasets-with-imblearn-library-df5e58b968f4).

Корий синтетично збалансував кількість класів у датасет в сторону збільшення.
```
sm = SMOTE(random_state=0)
X_resampled, y_resampled = sm.fit_resample(X, y)
```
```
DATASET f0s1. shape: (193860, 31)
class: 0, rows:   31170, idle   , prop: 0.1608
class: 1, rows:   55500, walking, prop: 0.2863
class: 2, rows:  102240, running, prop: 0.5274
class: 3, rows:    4950, stairs , prop: 0.02553
X.shape=(193860, 30)
X_resampled.shape=(408960, 30)
y_resampled.shape=(408960,)
class: 0, rows:  102240, idle   , prop: 0.25
class: 1, rows:  102240, walking, prop: 0.25
class: 2, rows:  102240, running, prop: 0.25
class: 3, rows:  102240, stairs , prop: 0.25
```
##### Classification report
```
limit_frames=None
--------------------------------------------------------------------------------
Data set: f0s0, shape: (193860, 4), model: SVC
              precision    recall  f1-score   support

        idle     0.9601    0.9839    0.9718      9351
     walking     0.8017    0.8979    0.8470     16650
     running     0.9279    0.9052    0.9164     30672
      stairs     1.0000    0.0054    0.0107      1485

    accuracy                         0.8928     58158
   macro avg     0.9224    0.6981    0.6865     58158
weighted avg     0.8988    0.8928    0.8823     58158

--------------------------------------------------------------------------------
Data set: f0s0, shape: (193860, 4), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     0.9997    1.0000    0.9998      9351
     walking     0.9995    0.9996    0.9995     16650
     running     0.9996    1.0000    0.9998     30672
      stairs     1.0000    0.9906    0.9953      1485

    accuracy                         0.9996     58158
   macro avg     0.9997    0.9975    0.9986     58158
weighted avg     0.9996    0.9996    0.9996     58158

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000     30672
     walking     1.0000    1.0000    1.0000     30672
     running     1.0000    1.0000    1.0000     30672
      stairs     1.0000    1.0000    1.0000     30672

    accuracy                         1.0000    122688
   macro avg     1.0000    1.0000    1.0000    122688
weighted avg     1.0000    1.0000    1.0000    122688

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: SVC_Linear
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000     30672
     walking     0.9939    0.9587    0.9760     30672
     running     1.0000    1.0000    1.0000     30672
      stairs     0.9601    0.9941    0.9768     30672

    accuracy                         0.9882    122688
   macro avg     0.9885    0.9882    0.9882    122688
weighted avg     0.9885    0.9882    0.9882    122688

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000     30672
     walking     1.0000    1.0000    1.0000     30672
     running     1.0000    1.0000    1.0000     30672
      stairs     1.0000    1.0000    1.0000     30672

    accuracy                         1.0000    122688
   macro avg     1.0000    1.0000    1.0000    122688
weighted avg     1.0000    1.0000    1.0000    122688

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1023
     walking     0.9851    0.9706    0.9778      1022
     running     1.0000    1.0000    1.0000      1023
      stairs     0.9711    0.9853    0.9781      1022

    accuracy                         0.9890      4090
   macro avg     0.9890    0.9890    0.9890      4090
weighted avg     0.9890    0.9890    0.9890      4090

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: SVC_Linear
              precision    recall  f1-score   support

        idle     0.9894    1.0000    0.9947      1023
     walking     0.8653    0.9178    0.8908      1022
     running     1.0000    0.9892    0.9946      1023
      stairs     0.9125    0.8571    0.8840      1022

    accuracy                         0.9411      4090
   macro avg     0.9418    0.9410    0.9410      4090
weighted avg     0.9418    0.9411    0.9410      4090

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1023
     walking     0.9941    0.9902    0.9922      1022
     running     1.0000    1.0000    1.0000      1023
      stairs     0.9903    0.9941    0.9922      1022

    accuracy                         0.9961      4090
   macro avg     0.9961    0.9961    0.9961      4090
weighted avg     0.9961    0.9961    0.9961      4090

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1023
     walking     0.9990    0.9765    0.9876      1022
     running     1.0000    1.0000    1.0000      1023
      stairs     0.9770    0.9990    0.9879      1022

    accuracy                         0.9939      4090
   macro avg     0.9940    0.9939    0.9939      4090
weighted avg     0.9940    0.9939    0.9939      4090

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: SVC_Linear
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1023
     walking     0.9940    0.9755    0.9847      1022
     running     1.0000    1.0000    1.0000      1023
      stairs     0.9760    0.9941    0.9850      1022

    accuracy                         0.9924      4090
   macro avg     0.9925    0.9924    0.9924      4090
weighted avg     0.9925    0.9924    0.9924      4090

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1023
     walking     0.9980    1.0000    0.9990      1022
     running     1.0000    1.0000    1.0000      1023
      stairs     1.0000    0.9980    0.9990      1022

    accuracy                         0.9995      4090
   macro avg     0.9995    0.9995    0.9995      4090
weighted avg     0.9995    0.9995    0.9995      4090
```
#### RandomUnderSampler
Додано додковий reshahping з викорисанням [RandomUnderSampler](https://medium.com/thecyphy/handling-imbalanced-datasets-with-imblearn-library-df5e58b968f4).

Корий синтетично збалансував кількість класів у датасет в сторону зменшення.
```
rundersampler = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rundersampler.fit_resample(X, y)
```
```
DATASET f1s0. shape: (6462, 91)
class: 0, rows:    1039, idle   , prop: 0.1608
class: 1, rows:    1850, walking, prop: 0.2863
class: 2, rows:    3408, running, prop: 0.5274
class: 3, rows:     165, stairs , prop: 0.02553
X.shape=(6462, 90)
X_resampled.shape=(660, 90)
y_resampled.shape=(660,)
class: 0, rows:     165, idle   , prop: 0.25
class: 1, rows:     165, walking, prop: 0.25
class: 2, rows:     165, running, prop: 0.25
class: 3, rows:     165, stairs , prop: 0.25
```
##### Classification report
```
limit_frames=None
--------------------------------------------------------------------------------
Data set: f0s0, shape: (193860, 4), model: SVC
              precision    recall  f1-score   support

        idle     0.9601    0.9839    0.9718      9351
     walking     0.8017    0.8979    0.8470     16650
     running     0.9279    0.9052    0.9164     30672
      stairs     1.0000    0.0054    0.0107      1485

    accuracy                         0.8928     58158
   macro avg     0.9224    0.6981    0.6865     58158
weighted avg     0.8988    0.8928    0.8823     58158

--------------------------------------------------------------------------------
Data set: f0s0, shape: (193860, 4), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     0.9997    1.0000    0.9998      9351
     walking     0.9995    0.9994    0.9994     16650
     running     0.9995    0.9999    0.9997     30672
      stairs     1.0000    0.9906    0.9953      1485

    accuracy                         0.9996     58158
   macro avg     0.9997    0.9975    0.9986     58158
weighted avg     0.9996    0.9996    0.9996     58158

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1485
     walking     1.0000    0.9778    0.9888      1485
     running     1.0000    1.0000    1.0000      1485
      stairs     0.9783    1.0000    0.9890      1485

    accuracy                         0.9944      5940
   macro avg     0.9946    0.9944    0.9944      5940
weighted avg     0.9946    0.9944    0.9944      5940

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: SVC_Linear
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1485
     walking     0.9882    0.9569    0.9723      1485
     running     1.0000    1.0000    1.0000      1485
      stairs     0.9582    0.9886    0.9732      1485

    accuracy                         0.9864      5940
   macro avg     0.9866    0.9864    0.9864      5940
weighted avg     0.9866    0.9864    0.9864      5940

--------------------------------------------------------------------------------
Data set: f0s1, shape: (193860, 31), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000      1485
     walking     1.0000    1.0000    1.0000      1485
     running     1.0000    1.0000    1.0000      1485
      stairs     1.0000    1.0000    1.0000      1485

    accuracy                         1.0000      5940
   macro avg     1.0000    1.0000    1.0000      5940
weighted avg     1.0000    1.0000    1.0000      5940

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000        50
     walking     0.8600    0.8776    0.8687        49
     running     1.0000    1.0000    1.0000        50
      stairs     0.8750    0.8571    0.8660        49

    accuracy                         0.9343       198
   macro avg     0.9337    0.9337    0.9337       198
weighted avg     0.9344    0.9343    0.9343       198

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: SVC_Linear
              precision    recall  f1-score   support

        idle     0.9804    1.0000    0.9901        50
     walking     0.7619    0.6531    0.7033        49
     running     1.0000    0.9800    0.9899        50
      stairs     0.6964    0.7959    0.7429        49

    accuracy                         0.8586       198
   macro avg     0.8597    0.8572    0.8565       198
weighted avg     0.8610    0.8586    0.8579       198

--------------------------------------------------------------------------------
Data set: f1s0, shape: (6462, 91), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000        50
     walking     0.8864    0.7959    0.8387        49
     running     1.0000    1.0000    1.0000        50
      stairs     0.8148    0.8980    0.8544        49

    accuracy                         0.9242       198
   macro avg     0.9253    0.9235    0.9233       198
weighted avg     0.9260    0.9242    0.9240       198

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: SVC
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000        50
     walking     0.9375    0.9184    0.9278        49
     running     1.0000    1.0000    1.0000        50
      stairs     0.9200    0.9388    0.9293        49

    accuracy                         0.9646       198
   macro avg     0.9644    0.9643    0.9643       198
weighted avg     0.9647    0.9646    0.9646       198

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: SVC_Linear
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000        50
     walking     0.8462    0.8980    0.8713        49
     running     1.0000    1.0000    1.0000        50
      stairs     0.8913    0.8367    0.8632        49

    accuracy                         0.9343       198
   macro avg     0.9344    0.9337    0.9336       198
weighted avg     0.9350    0.9343    0.9343       198

--------------------------------------------------------------------------------
Data set: f1s1, shape: (6462, 118), model: RandomForestClassifier
              precision    recall  f1-score   support

        idle     1.0000    1.0000    1.0000        50
     walking     1.0000    0.9796    0.9897        49
     running     1.0000    1.0000    1.0000        50
      stairs     0.9800    1.0000    0.9899        49

    accuracy                         0.9949       198
   macro avg     0.9950    0.9949    0.9949       198
weighted avg     0.9951    0.9949    0.9949       198
```
