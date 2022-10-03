from .middleware import ArrivalMiddleware

def main():
    mid = ArrivalMiddleware()

    mid.start_threads()

    mid.join()