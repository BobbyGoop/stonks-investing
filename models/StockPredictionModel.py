from math import sqrt

import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


class StockPredictionModel:
	MODEL_INPUT_DAYS = 56
	PREDICTION_INPUT_DAYS = 28

	def __init__(self, stock, start_date, end_date, plot=True):
		self.stock_df = yf.download(stock, start=start_date, end=end_date, progress=False).loc[:, ['Close']]
		self.model = Sequential([
			layers.Input((self.MODEL_INPUT_DAYS, 1)),
			layers.LSTM(64),
			layers.Dense(32, activation='relu'),
			layers.Dense(32, activation='relu'),
			layers.Dense(1)
		])

		self.plot = plot
		self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

	def _window_data(self, n=MODEL_INPUT_DAYS):
		windowed_df = pd.DataFrame()
		for i in range(n, 0, -1):
			windowed_df[f'Target-{i}'] = self.stock_df['Close'].shift(i)
		windowed_df['Target'] = self.stock_df['Close']
		windowed_df = windowed_df.dropna()

		dates_list = np.array(windowed_df.index)
		windowed_df = windowed_df.to_numpy()

		middle_matrix = windowed_df[:, 0:-1]
		X_values = middle_matrix.reshape((len(dates_list), middle_matrix.shape[1], 1))
		Y_values = windowed_df[:, -1]

		return dates_list, X_values.astype(np.float32), Y_values.astype(np.float32)

	def train_model(self, epochs=100, batch_size=32, train_percentage=0.6, val_percentage=0.2):
		self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

		dates, X, y = self._window_data()

		index_train = int(len(dates) * train_percentage)
		index_validation = int(len(dates) * (train_percentage + val_percentage))

		dates_train, X_train, y_train = dates[:index_train + 1], X[:index_train + 1], y[:index_train + 1]
		dates_val, X_val, y_val = dates[index_train:index_validation], X[index_train:index_validation], y[index_train:index_validation]
		dates_test, X_test, y_test = dates[index_validation:], X[index_validation:], y[index_validation:]

		self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

		train_predictions = self.model.predict(X_train).flatten()
		val_predictions = self.model.predict(X_val).flatten()
		test_predictions = self.model.predict(X_test).flatten()
		self.model.summary()

		if self.plot:
			fig, ax = plt.subplots()
			fig.suptitle(f"Обучение модели LSTM ({len(dates_train)} значений)")
			ax.plot(dates_train, train_predictions)
			ax.plot(dates_train, y_train, linestyle="dashed")
			ax.plot(np.append(dates_val, dates_test), np.append(val_predictions, test_predictions), linestyle="dotted")
			ax.plot(np.append(dates_val, dates_test), np.append(y_val, y_test), linestyle="dotted")
			ax.axvline(x=dates_train[-1], color='b', label='Граница обучающего и тестового множеств')
			ax.legend(['Отклик модели', 'Действительные значения'])
			ax.set_xlabel("Дата")
			ax.set_ylabel("Цена")
			ax.xaxis.set_major_formatter(self.date_format)
			fig.autofmt_xdate()
			plt.subplots_adjust(left=0.04, bottom=0.1, right=0.96, top=0.9)
			plt.grid()

	def predict_values_recursively(self):
		dates, X, y = self._window_data()
		# Assume we want to predict next 14 days
		# while model is trained across 28 days
		recursive_predictions = []
		recursive_dates = dates[-self.PREDICTION_INPUT_DAYS:]
		last_window = X[-self.PREDICTION_INPUT_DAYS]

		for _ in recursive_dates:
			# print(last_window)
			next_prediction = self.model.predict(np.array([last_window])).flatten()
			recursive_predictions.append(next_prediction)
			new_window = list(last_window[1:])
			new_window.append(next_prediction)
			new_window = np.array(new_window)
			last_window = new_window

		# Count metrics
		MSE = mean_squared_error(y[-self.PREDICTION_INPUT_DAYS:], recursive_predictions)
		RMSE = sqrt(MSE)
		MAE = mean_absolute_error(y[-self.PREDICTION_INPUT_DAYS:], recursive_predictions)
		print("Mean Squared Error: ", MSE)
		print("Root Mean Squared Error: ", RMSE)
		print("Mean Absolute Error: ", MAE)

		if self.plot:
			fig, ax = plt.subplots()
			fig.suptitle(f"Предсказание значений")
			ax.plot(dates[-self.PREDICTION_INPUT_DAYS:], y[-self.PREDICTION_INPUT_DAYS:], linestyle="dotted")
			ax.plot(recursive_dates, recursive_predictions)
			ax.set_xticks(recursive_dates)
			ax.legend(['Отклик модели (по известным данным)', 'Реальные значения', 'Прогнозируемые значения'])
			ax.set_xlabel("Дата")
			ax.set_ylabel("Цена")
			ax.xaxis.set_major_formatter(self.date_format)
			fig.autofmt_xdate()
			plt.subplots_adjust(left=0.04, bottom=0.1, right=0.96, top=0.9)
			plt.grid()

	def show_plots(self):
		if self.plot:
			plt.show()
		return self.stock_df
