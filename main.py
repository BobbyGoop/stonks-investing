import datetime

from dateutil.relativedelta import relativedelta

from models.StockPredictionModel import StockPredictionModel
from models.TradingHelper import TradingHelper

if __name__ == "__main__":
	end_date = datetime.date(day=1, month=1, year=2022)
	start_date = end_date + relativedelta(years=-2)
	stock = "AAPL"

	ltsm_model = StockPredictionModel(stock, start_date, end_date)
	trade_helper = TradingHelper(stock, start_date, end_date)

	trade_helper.count_levels()
	trade_helper.count_esma()
	trade_helper.stock_updates()
	trade_helper.export_data()
	trade_helper.show_plots()

	ltsm_model.train_model()
	ltsm_model.predict_values_recursively()
	ltsm_model.show_plots()
