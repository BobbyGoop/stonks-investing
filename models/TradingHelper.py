import datetime as dt

import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf


class TradingHelper:
	caching_size = 20
	stable_area = 15

	def __init__(self, stock, start_date, end_date, plotting=True):

		self.stock = stock
		self.plotting = plotting
		self.start_date = start_date
		self.end_date = end_date
		self.stock_data = yf.download(stock, start=self.start_date, end=self.end_date, progress=False)

	def count_esma(self):
		# Count SMA30 / SMA90
		SMA30 = pd.DataFrame()
		SMA30['Close Price'] = self.stock_data['Close'].rolling(window=30).mean()
		SMA90 = pd.DataFrame()
		SMA90['Close Price'] = self.stock_data['Close'].rolling(window=90).mean()
		data = pd.DataFrame()
		data['Stock'] = self.stock_data['Close']
		data['SMA30'] = SMA30['Close Price']
		data['SMA90'] = SMA90['Close Price']

		# Count EMA20 / EMA60
		EMA20 = pd.DataFrame()
		EMA20['Close Price'] = self.stock_data['Close'].ewm(span=20).mean()
		EMA60 = pd.DataFrame()
		EMA60['Close Price'] = self.stock_data['Close'].ewm(span=60).mean()
		data = pd.DataFrame()
		data['Stock'] = self.stock_data['Close']
		data['EMA20'] = EMA20['Close Price']
		data['EMA60'] = EMA60['Close Price']
		if self.plotting:
			fig, (ema, sma) = plt.subplots(2, 1)
			fig.suptitle("SMA и EMA сглаживание")
			# Визуализируем
			sma.plot(data['Stock'], label=self.stock, alpha=0.35)
			sma.plot(SMA30['Close Price'], label='SMA30')
			sma.plot(SMA90['Close Price'], label='SMA90')
			# sma.set_xlabel('01/01/2019 - ' + current_date)
			sma.set_ylabel('Цена закрытия')
			sma.set_xlabel('Дата')
			sma.legend(loc='upper left')

			# Визуализируем
			ema.plot(data['Stock'], label=self.stock, alpha=0.35)
			ema.plot(EMA20['Close Price'], label='EMA30')
			ema.plot(EMA60['Close Price'], label='EMA60')
			# ema.set_xlabel('01/01/2019 - ' + current_date)
			ema.axes.get_xaxis().set_visible(False)
			ema.set_ylabel('Цена закрытия')
			ema.legend(loc='upper left')

			# Formatting Date
			date_format = mpl_dates.DateFormatter('%d-%m-%Y')
			sma.xaxis.set_major_formatter(date_format)
			ema.xaxis.set_major_formatter(date_format)
			fig.autofmt_xdate()

			plt.subplots_adjust(left=0.04, bottom=0.1, right=0.96, top=0.9, wspace=0.05, hspace=0.0)
		return [(SMA30, SMA90), (EMA20, EMA60)]

	def get_max_pp(self):
		pivots = []
		dates = []
		counter = 0

		cache = [0] * self.caching_size
		date_range = [0] * self.caching_size

		for i in self.stock_data.index:
			currentMax = max(cache, default=0)
			value = round(self.stock_data['High'][i], 2)

			cache.pop(0)
			date_range.pop(0)
			cache.append(value)
			date_range.append(i)

			if currentMax == max(cache, default=0):
				counter += 1
			else:
				counter = 0
			if counter == self.stable_area:
				pivots.append(currentMax)
				dates.append(date_range[cache.index(currentMax)])
		return pivots, dates

	def get_min_pp(self):
		pivots = []
		dates = []
		counter = 0

		cache = [99999] * self.caching_size
		date_range = [0] * self.caching_size

		for i in self.stock_data.index:
			currentMin = min(cache, default=0)
			value = round(self.stock_data['Low'][i], 2)

			cache.pop(0)
			date_range.pop(0)
			cache.append(value)
			date_range.append(i)

			if currentMin == min(cache, default=0):
				counter += 1
			else:
				counter = 0
			if counter == self.stable_area:
				pivots.append(currentMin)
				dates.append(date_range[cache.index(currentMin)])
		return pivots, dates

	def count_levels(self):
		min_pivots, min_dates = self.get_min_pp()
		max_pivots, max_dates = self.get_max_pp()
		delta = dt.timedelta(days=self.stable_area)
		if self.plotting:

			fig, ax = plt.subplots()
			# plt.figure(figsize=(12.6, 4.6))
			ax.set_title(self.stock + ' history')
			ax.set_xlabel("From " + self.start_date.strftime("%d.%m.%Y") + " to " + self.end_date.strftime("%d.%m.%Y"))
			ax.set_ylabel('Close price')

			for i in self.stock_data.index:
				ax.plot([i, i], [self.stock_data["Low"][i], self.stock_data["High"][i]], c="b", alpha=0.4)
			ax.plot([], [], label="High-Low Difference Daily")

			for index in range(len(min_pivots)):
				ax.plot_date([min_dates[index], min_dates[index] + delta], [min_pivots[index], min_pivots[index]], linestyle='-', linewidth=2, c="g")
			ax.plot([], [], label="Max Pivot Points", c="g")

			for index in range(len(max_pivots)):
				ax.plot_date([max_dates[index], max_dates[index] + delta], [max_pivots[index], max_pivots[index]], linestyle='-', linewidth=2, c="r")
			ax.plot([], [], label="Min Pivot Points", c="r")

			# Formatting Date
			date_format = mpl_dates.DateFormatter('%d-%m-%Y')
			ax.xaxis.set_major_formatter(date_format)
			fig.autofmt_xdate()
			plt.legend(loc='best')
		return [(max_pivots, max_pivots), (min_dates, max_dates)]

	def stock_updates(self):
		df = self.stock_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
		df.index = pd.to_datetime(df.index)

		ticks = np.array(df.index.astype(np.int64))
		price_close = np.array(df['Close'].to_list())

		trend_close = np.polyfit(ticks, price_close, 1)
		trend_close = np.poly1d(trend_close)

		fig, ax = plt.subplots()
		mpf.plot(df, type='candlestick', style='yahoo', show_nontrading=True, ax=ax)
		ax.set_xlabel('Дата')
		ax.set_ylabel('Цена')
		ax.plot(df.index, trend_close(ticks), label="Тренд")
		fig.suptitle(f'Изменение цены акции {self.stock}')

		# Formatting Date
		date_format = mpl_dates.DateFormatter('%d-%m-%Y')
		ax.xaxis.set_major_formatter(date_format)
		fig.autofmt_xdate()

		plt.subplots_adjust(left=0.04, bottom=0.1, right=0.96, top=0.9)
		plt.legend(loc="best")
		plt.grid()

	def show_plots(self):
		plt.show()
		return self.stock_data

	def export_data(self, file_name=""):
		file_name = f"data/{self.stock}_Quotes.csv" if not file_name else ""
		self.stock_data.to_csv(file_name, sep=";")
