import datetime as dt
from datetime import date

import investpy
import matplotlib
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TradingHelper:
	caching_size = 40
	stable_area = 35

	def __init__(self, stock, country, start_date=dt.date(2021, 1, 1), end_date=date.today(), plotting=True):
		self.stock = stock
		self.country = country
		self.plotting = plotting
		self.start_date = start_date
		self.end_date = end_date
		self.stock_data = yf.download(stock, start=start_date, end=end_date, progress=False)

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
			fig.suptitle(stock + ' history (SMA and EMA)')
			# Визуализируем
			sma.plot(data['Stock'], label=stock, alpha=0.35)
			sma.plot(SMA30['Close Price'], label='SMA30')
			sma.plot(SMA90['Close Price'], label='SMA90')
			# sma.set_xlabel('01/01/2019 - ' + current_date)
			sma.set_ylabel('Close price')
			sma.legend(loc='upper left')

			# Визуализируем
			ema.plot(data['Stock'], label=stock, alpha=0.35)
			ema.plot(EMA20['Close Price'], label='EMA30')
			ema.plot(EMA60['Close Price'], label='EMA60')
			# ema.set_xlabel('01/01/2019 - ' + current_date)
			ema.axes.get_xaxis().set_visible(False)
			ema.set_ylabel('Close price')
			ema.legend(loc='upper left')
			plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.0)
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
		lastPivot = 0

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

			plt.figure(figsize=(12.6, 4.6))
			plt.title(stock + ' history')
			plt.xlabel("From " + self.start_date.strftime("%d.%m.%Y") + " to " + self.end_date.strftime("%d.%m.%Y"))
			plt.ylabel('Close price')

			for i in self.stock_data.index:
				plt.plot([i, i], [self.stock_data["Low"][i], self.stock_data["High"][i]], c="b", alpha=0.4)
			plt.plot([], [], label="High-Low Difference Daily")

			for index in range(len(min_pivots)):
				plt.plot_date([min_dates[index], min_dates[index] + delta], [min_pivots[index], min_pivots[index]], linestyle='-', linewidth=2, c="g", marker=",")
			plt.plot([], [], label="Max Pivot Points", c="g")

			for index in range(len(max_pivots)):
				plt.plot_date([max_dates[index], max_dates[index] + delta], [max_pivots[index], max_pivots[index]], linestyle='-', linewidth=2, c="r", marker=",")
			plt.plot([], [], label="Min Pivot Points", c="r")

			plt.legend(loc='best')

		return [(max_pivots, max_pivots), (min_dates, max_dates)]

	def stock_updates(self):
		current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)

		plt.figure(figsize=(12.6, 4.6))
		plt.plot(self.stock_data['Open'], label=stock + " Open price", alpha=1, c="green")
		plt.plot(self.stock_data['Close'], label=stock + " Close price", alpha=1, c="orange")
		plt.title(stock + ' history')
		plt.xlabel('Date')
		plt.ylabel('Close price')
		plt.legend(loc='upper left')
		plt.show()
		# print('Prices Last Five days of ' + stock + ' =', np.array(df['Close'][-5:][0]), ';', np.array(df['Close'][-5:][1]), ';', np.array(df['Close'][-5:][2]), ';', np.array(df['Close'][-5:][3]), ';', np.array(df['Close'][-5:][4]))
		# p_1 = abs(1 - df['Close'][-5:][1] / df['Close'][-5:][0])
		# if df['Close'][-5:][1] >= df['Close'][-5:][0]:
		# 	pp_1 = '+' + str(round(p_1 * 100, 2)) + '%'
		# else:
		# 	pp_1 = '-' + str(round(p_1 * 100, 2)) + '%'
		# p_2 = abs(1 - df['Close'][-5:][2] / df['Close'][-5:][1])
		# if df['Close'][-5:][2] >= df['Close'][-5:][1]:
		# 	pp_2 = '+' + str(round(p_2 * 100, 2)) + '%'
		# else:
		# 	pp_2 = '-' + str(round(p_2 * 100, 2)) + '%'
		# p_3 = abs(1 - df['Close'][-5:][3] / df['Close'][-5:][2])
		# if df['Close'][-5:][3] >= df['Close'][-5:][2]:
		# 	pp_3 = '+' + str(round(p_3 * 100, 2)) + '%'
		# else:
		# 	pp_3 = '-' + str(round(p_3 * 100, 2)) + '%'
		# p_4 = abs(1 - df['Close'][-5:][4] / df['Close'][-5:][3])
		# if df['Close'][-5:][4] >= df['Close'][-5:][3]:
		# 	pp_4 = '+' + str(round(p_4 * 100, 2)) + '%'
		# else:
		# 	pp_4 = '-' + str(round(p_4 * 100, 2)) + '%'
		# print('Percentage +/- of ' + stock + ' =', pp_1, ';', pp_2, ';', pp_3, ';', pp_4, )

	def show_plots(self):
		plt.show()
		return self.stock_data

if __name__ == "__main__":
	plt.style.use('dark_background')
	stock = 'AAPL'
	country = 'United States'
	matplotlib.get_backend()
	trader = TradingHelper(stock, country)

	trader.count_levels()
	trader.count_esma()
	trader.stock_updates()
	trader.show_plots()
	# count_esma(stock, country)
	# Stock_EMA(stock, country)
	# Upper_levels(stock, country)
	# Low_levels(stock, country)
	# Last_Month(stock, country)
	# count_levels(stock, country)