from multiprocessing import Value

# get dynamic max log fft values
analyze_log_fft_max_value = Value('f', 0)
analyze_log_piano_key_max_value = Value('f', 0)
