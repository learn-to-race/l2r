from middleware import ArrivalMiddleware

mid = ArrivalMiddleware()

mid.start_threads()

mid.join()