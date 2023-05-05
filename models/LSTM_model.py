import yfinance as yf
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

if __name__ == "__main__":

    AAPL = yf.download('AAPL',
                       start='2015-01-01',
                       end='2020-06-06',
                       progress=False)

    # Создаем новый датафрейм только с колонкой "Close"
    data = AAPL.filter(['Close'])
    # преобразовываем в нумпаевский массив
    dataset = data.values
    # Вытаскиваем количество строк в дате для обучения модели (LSTM)
    training_data_len = math.ceil(len(data.values) * .6)

    train_dataset = data.values[0:training_data_len]
    test_dataset = data.values[training_data_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_dataset = scaler.fit_transform(train_dataset)
    test_dataset = scaler.fit_transform(test_dataset)

    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    steps = 3
    for i in range(steps, len(train_dataset)):
        x_train.append(train_dataset[i - steps:i])
        y_train.append(train_dataset[i])

    for i in range(steps, len(test_dataset)):
        x_test.append(test_dataset[i - steps:i])
        y_test.append(test_dataset[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    shapes = (x_train.shape[1], x_train.shape[2])

    # Строим нейронку
    model = Sequential()
    model.add(LSTM(4, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))

    # Компилируем и Тренируем модель
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=16, epochs=4, verbose=1, validation_data=(x_test, y_test))

    model.summary()

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    # train_predict = np.c_[train_predict, np.zeros(train_predict)]
    # test_predict = np.c_[test_predict, np.zeros(test_predict)]

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)


    # # Получаем модель предсказывающую значения
    # predictions = model.predict(x_test)
    # predictions = scaler.inverse_transform(predictions)
    #
    # # Получим mean squared error (RMSE) - метод наименьших квадратов
    # rms_error= np.sqrt(np.mean(predictions - y_test) ** 2)

    # Строим график
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = test_predict
    # Визуализируем
    plt.figure(figsize=(16, 8))
    plt.title('Model LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
    plt.show()
