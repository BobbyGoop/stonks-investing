import datetime
from math import sqrt

import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_INPUT_DAYS = 56
PREDICTION_INPUT_DAYS = 28


def str_to_datetime(s):
	split = s.split('-')
	year, month, day = int(split[0]), int(split[1]), int(split[2])
	return datetime.datetime(year=year, month=month, day=day)


def window_data(data, n=MODEL_INPUT_DAYS):
	windowed_data = pd.DataFrame()
	for i in range(n, 0, -1):
		windowed_data[f'Target-{i}'] = data['Close'].shift(i)
	windowed_data['Target'] = data['Close']
	return windowed_data.dropna()


def windowed_df_to_date_xy(windowed_dataframe):
	df_as_np = windowed_dataframe.to_numpy()
	dates_list = np.array(windowed_dataframe.index)

	middle_matrix = df_as_np[:, 0:-1]
	X_values = middle_matrix.reshape((len(dates_list), middle_matrix.shape[1], 1))

	# print(X[:5, :])
	Y_values = df_as_np[:, -1]

	return dates_list, X_values.astype(np.float32), Y_values.astype(np.float32)


if __name__ == "__main__":

	end_date = datetime.date(day=1, month=1, year=2022)
	start_date = end_date + relativedelta(years=-2)
	AAPL = yf.download('YNDX', start=start_date, end=end_date, progress=False)
	# print(AAPL.columns)
	df = AAPL.loc[:, ['Close']]
	# print(df)
	# Start day second time around: '2021-03-25'
	windowed_df = window_data(df)
	# print(windowed_df)
	dates, X, y = windowed_df_to_date_xy(windowed_df)

	# print(dates[:10])
	# print(X[:10, :])
	# print(y)
	q_80 = int(len(dates) * .6)
	q_90 = int(len(dates) * .8)

	dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

	dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
	dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

	model = Sequential([layers.Input((MODEL_INPUT_DAYS, 1)), layers.LSTM(64), layers.Dense(32, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(1)])

	model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

	train_predictions = model.predict(X_train).flatten()
	val_predictions = model.predict(X_val).flatten()
	test_predictions = model.predict(X_test).flatten()

	# Assume we want to predict next 14 days
	# while model is trained across 28 days
	recursive_predictions = []
	recursive_dates = dates_test[-PREDICTION_INPUT_DAYS:]
	last_window = X_test[-PREDICTION_INPUT_DAYS]

	for target_date in recursive_dates:
		# print(last_window)
		next_prediction = model.predict(np.array([last_window])).flatten()
		recursive_predictions.append(next_prediction)
		new_window = list(last_window[1:])
		new_window.append(next_prediction)
		new_window = np.array(new_window)
		last_window = new_window

	model.summary()
	# Mean Squared Error
	MSE = mean_squared_error(y_test[-PREDICTION_INPUT_DAYS:], recursive_predictions)
	RMSE = sqrt(MSE)
	MAE = mean_absolute_error(y_test[-PREDICTION_INPUT_DAYS:], recursive_predictions)
	print("Mean Squared Error: ", MSE)
	print("Root Mean Squared Error: ", RMSE)
	print("Mean Absolute Error: ", MAE)

	# Plotting data
	date_format = mpl_dates.DateFormatter('%d-%m-%Y')

	fig, ax = plt.subplots()
	fig.suptitle(f"Обучение модели LSTM ({len(dates_train)} значений)")
	ax.plot(dates_train, train_predictions)
	ax.plot(dates_train, y_train, linestyle="dashed")
	ax.legend(['Отклик модели', 'Действительные значения'])
	ax.set_xlabel("Дата")
	ax.set_ylabel("Цена")
	ax.xaxis.set_major_formatter(date_format)
	fig.autofmt_xdate()
	plt.tight_layout()
	plt.grid()

	fig, ax = plt.subplots()
	fig.suptitle(f"Валидация модели LSTM ({len(dates_val)} значений)")
	ax.plot(dates_val, val_predictions)
	ax.plot(dates_val, y_val, linestyle="dashed")
	ax.legend(['Отклик модели', 'Действительные значения'])
	ax.set_xlabel("Дата")
	ax.set_ylabel("Цена")
	ax.xaxis.set_major_formatter(date_format)
	fig.autofmt_xdate()
	plt.tight_layout()
	plt.grid()

	fig, ax = plt.subplots()
	fig.suptitle(f"Предсказание значений")
	ax.plot(dates_test[-PREDICTION_INPUT_DAYS:], test_predictions[-PREDICTION_INPUT_DAYS:], linestyle="dashdot")
	ax.plot(dates_test[-PREDICTION_INPUT_DAYS:], y_test[-PREDICTION_INPUT_DAYS:], linestyle="dashed")
	ax.plot(recursive_dates, recursive_predictions)
	ax.set_xticks(recursive_dates)
	ax.legend(['Отклик модели (по известным значениям)', 'Реальные значения', 'Действительные значения'])
	ax.set_xlabel("Дата")
	ax.set_ylabel("Цена")
	ax.xaxis.set_major_formatter(date_format)
	fig.autofmt_xdate()
	plt.tight_layout()
	plt.grid()

	plt.show()
