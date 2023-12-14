import psutil
import GPUtil
import csv
import multiprocessing
import threading
import time
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np



csv_file_path = "C:\\Users\\Jaggi Reddy\\Desktop\\cpu_info.csv"

def measure_cpu_performance():
    start_time = time.time()
    cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
    end_time = time.time()

    avg_cpu_percentage = sum(cpu_percentages) / len(cpu_percentages)
    elapsed_time = end_time - start_time


    return avg_cpu_percentage


def get_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    return cpu_count, cpu_usage

def get_memory_info():
    memory_info = psutil.virtual_memory()
    return memory_info

def get_storage_info():
    disk_info = psutil.disk_usage('/')
    return disk_info

def get_gpu_info():
    gpu_info = GPUtil.getGPUs()
    return gpu_info

def train_tf_model():
    # Generate a synthetic dataset
    num_samples = 10000
    input_shape = (32, 32, 3)

    x_train = np.random.random((num_samples, *input_shape))
    y_train = np.random.randint(2, size=(num_samples, 1))

    # Define a simple convolutional neural network
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the synthetic dataset
    batch_size = 64
    epochs = 4

    start_time = time.time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=0)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training time for TensorFlow model: {elapsed_time:.2f} seconds")

def intensive_calculation():
    start_time = time.time()
    result = 0
    for _ in range(10**8):  # Increased loop range
        result = (result + 1) * 1.001  # Intensive mathematical operation
    end_time = time.time()    
    print(f"Execution Time: {end_time - start_time} seconds")
    return result

def task1():
    for _ in range(2):
        print("Task 1 running...")
        train_tf_model()

def task2():
    for _ in range(2):
        print("Task 2 running...")
        intensive_calculation()

def multiprocessing_example():
    print("Multiprocessing example:")
    start_time = time.time()

    process1 = multiprocessing.Process(target=task1)
    process2 = multiprocessing.Process(target=task2)

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    elapsed_time = end_time - start_time
    return elapsed_time 

def multithreading_example():
    print("Multithreading example:")
    start_time = time.time()

    thread1 = threading.Thread(target=task1)
    thread2 = threading.Thread(target=task2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    elapsed_time = end_time - start_time
    return elapsed_time


def normal_example():
    print("Normal execution example:")
    start_time = time.time()
    task1()
    task2()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    elapsed_time = end_time - start_time
    return elapsed_time      


def main():
    cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
    # CPU Information
    cpu_count, cpu_usage = get_cpu_info()
    print("Number of Physical CPU Cores:", cpu_count)
    print("CPU Usage:", cpu_usage)

    # Memory Information
    memory_info = get_memory_info()
    print("Memory Info:", memory_info)

    # Storage Information
    storage_info = get_storage_info()
    print("Storage Info:", storage_info)

    # GPU Information
    gpu_info = get_gpu_info()
    print("GPU Info:", gpu_info)
   
    print("\n")
    mp=multiprocessing_example()
    print("\n")
    
    mt=multithreading_example()
    print("\n")

    print("\n")
    ne=normal_example()
    
    
    avg_cpu_percentage = sum(cpu_percentages) / len(cpu_percentages)
    # Use 'with' statement to ensure the file is properly closed after writing
    with open(csv_file_path, mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
    
        # Write the header

        csv_writer.writerow(['cpu_count',cpu_count])
        csv_writer.writerow(['Average CPU performance',avg_cpu_percentage])
        csv_writer.writerow(["Total Memory:", memory_info.total / (1024 ** 3)])
        csv_writer.writerow(["Used Memory: ", memory_info.used  / (1024 ** 3)])
        csv_writer.writerow(["Free Memory: ", memory_info.free  / (1024 ** 3)])

        
        csv_writer.writerow(["GPU Info:", gpu_info ])        
        csv_writer.writerow(['Multi processing time in sec',mp])
        csv_writer.writerow(['Multi Threading time in sec',mt])
        csv_writer.writerow(['Normal processing time in sec',ne])
        csv_writer.writerow([""])
        csv_writer.writerow(["Storage Info:",storage_info])
        


if __name__ == "__main__":
    
    print(f"CPU information has been written to {csv_file_path}")
    main()



if __name__ == "__main__":
    # Example: Run main() with different numbers of cores
    for cores in [2, 4, 6]:
        print(f"Running with {cores} CPU cores")
        main(cores)


