pipe = ForecastPipeline(config)

pipe.load_dataset(...)

pipe.train()

pipe.forecast()

pipe.evaluate()

pipe.estimate_rul()

pipe.show_dashboard()