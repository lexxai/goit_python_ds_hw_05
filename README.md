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
