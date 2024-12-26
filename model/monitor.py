# import time
# import logging
# from functools import wraps

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# def monitor_prediction_time():
#     """
#     Decorator factory to monitor the execution time of the prediction function.
#     Returns the actual decorator.
#     """
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             start_time = time.perf_counter()  # High precision timer
#             try:
#                 result = func(*args, **kwargs)  # Call the original function
#             finally:
#                 end_time = time.perf_counter()
#                 execution_time = end_time - start_time
#                 logging.info(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
#             return result
#         return wrapper
#     return decorator

import time
from functools import wraps

def monitor_prediction_time(func):
    """
    Decorator to monitor the execution time of the prediction function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        
        execution_time = end_time - start_time  # Calculate the time difference
        print(f"Prediction time: {execution_time:.4f} seconds")  # Print the time taken
        
        return result  # Return the result of the original function
    
    return wrapper