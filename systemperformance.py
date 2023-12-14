import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import multiprocessing
import threading
import csv
import psutil  # To monitor CPU and memory usage
import os
import torch
k=0
# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
    k=1
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")
    


def set_affinity_to_n_physical_cores(n):
    try:
        # Get the current process ID
        pid = os.getpid()

        # Get the number of physical cores
        physical_cores = psutil.cpu_count(logical=False)

        # Ensure n is within the valid range
        if 1 <= n <= physical_cores:
            # Set the CPU affinity for the process to the first n physical cores
            target_cores = list(range(n))
            psutil.Process(pid).cpu_affinity(target_cores)

            print(f"Process affinity set to {n} physical cores: {target_cores}")
        else:
            print(f"Invalid value for n. Please choose a value between 1 and {physical_cores}.")
    except Exception as e:
        print(f"Error: {e}")



    
def task1(): #LSTM
    #LSTM for stock prediction
    start_time = time.time()
    # Load the CSV file
    file_path = r"C:\Users\NithinNandelli\Downloads\stock.csv" # Replace with your CSV file path
    stock_data = pd.read_csv(file_path)
    
    # Convert 'Time' to datetime and sort the data
    stock_data['Time'] = pd.to_datetime(stock_data['Time'])
    stock_data.sort_values('Time', inplace=True)
    
    # Normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
    
    # Function to create features and labels
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0:-1]  # all features except 'Close'
            dataX.append(a)
            dataY.append(dataset[i + time_step, -1])  # 'Close' as the label
        return np.array(dataX), np.array(dataY)
    
    # Define the time step and create dataset
    time_step = 60  # Using 60 time steps to predict the next step
    X, y = create_dataset(scaled_data, time_step)
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
    
    # Building the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 4)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=6)
    
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    
    
    # Fit a separate scaler for 'Close' prices
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = stock_data[['Close']].values
    close_scaler.fit(close_prices)
    
    # Making predictions for the next 10 steps
    last_sequence = X[-1]
    next_predictions = []
    
    for _ in range(10):
        last_sequence_reshaped = last_sequence.reshape((1, time_step, 4))
        next_value = model.predict(last_sequence_reshaped)[0, 0]
        next_predictions.append(next_value)
    
        # Update the sequence with the prediction
        new_row = np.array([last_sequence[-1, 1], last_sequence[-1, 2], last_sequence[-1, 3], next_value])
        last_sequence = np.vstack((last_sequence[1:], new_row))
    
    # Transform the predictions back to the original scale using the close_scaler
    next_predictions_scaled = close_scaler.inverse_transform(
        np.array(next_predictions).reshape(-1, 1)
    )
    
    print("Next 10 Close Values Predictions:", next_predictions_scaled.reshape(-1))
    end_time = time.time() 
    timelstm = end_time - start_time
    print(f"Execution Time of LSTM: {end_time - start_time} seconds")
    return timelstm

def task2():
    start_time = time.time()
    result = 0
    for _ in range(10**8):  # Increased loop range
        result = (result + 1) * 1.001  # Intensive mathematical operation
    end_time = time.time()
    timemath = end_time - start_time    
    print(f"Execution Time: {end_time - start_time} seconds")
    return timemath

def run_sequential():
    start_time = time.time()
    tt1=task1()
    tt2=task2()
    end_time = time.time()
    timeseq = end_time - start_time
    return tt1,tt2,timeseq
    
def run_multi_threading():
    
    start_time = time.time()
    thread1 = threading.Thread(target=task1)
    thread2 = threading.Thread(target=task2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
    
    end_time = time.time()
    timemthread = end_time - start_time
    return timemthread

def run_multi_processing():
    start_time = time.time()
    process1 = multiprocessing.Process(target=task1)
    process2 = multiprocessing.Process(target=task2)

    process1.start()
    process2.start()

    process1.join()
    process2.join()
    end_time = time.time()
    timemprocess = end_time - start_time
    return timemprocess    
    

def measure_performance():
    # Measure CPU and memory usage
    cpu_percent = psutil.cpu_percent()
    memory_used = psutil.virtual_memory().used

    return cpu_percent, memory_used

def write_results_to_csv(file_path, results):
    fieldnames = ["Cores","GPU", "Task1 Execution Time", 
                  "Task2 Execution Time", "Sequential Execution time", 
                  "Multi-thread Execution time","Multi-process Execution time",
                  "Total Execution Time", "CPU Percent", "Memory Used"]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        for result in results:
            writer.writerow(result)

def main():
    cores = [1,2,3,4,5,6,7,8]
    
    use_gpu = [False, True] if k == 1 else [False]

    results = []

    for core in cores:
        for gpu in use_gpu:
            device = torch.device("cpu")
            
            if gpu:
                device = torch.device("cuda")
             
                pass
            print(f"\nRunning with {core} core(s) and GPU: {gpu}")

            set_affinity_to_n_physical_cores(core)
            
            start_time = time.time()

            # Choose the appropriate run function based on core and GPU settings
            
            tmult=run_multi_threading()
            tprocess=run_multi_processing()
            tt1,tt2,timeseq=run_sequential()

            execution_time = time.time() - start_time

            cpu_percent, memory_used = measure_performance()

            results.append({
                "Cores": core,
                "GPU": gpu,
                "Task1 Execution Time":tt1,
                "Task2 Execution Time":tt2,
                "Sequential Execution time":timeseq,
                "Multi-thread Execution time":tmult,
                "Multi-process Execution time":tprocess,
                "Total Execution Time": execution_time,
                "CPU Percent": cpu_percent,
                "Memory Used": memory_used
            })
    # Write results to CSV file
    csv_file_path = r"C:\Users\NithinNandelli\Downloads\Anirudhproject\results.csv"
    write_results_to_csv(csv_file_path, results)


if __name__ == "__main__":
    main()





file_path = r"C:\Users\NithinNandelli\Downloads\Anirudhproject\results.csv
df = pd.read_csv(file_path)

# Separate data for GPU and non-GPU configurations
gpu_data = df[df['GPU'] == True]
cpu_data = df[df['GPU'] == False]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
opacity = 0.8

# Bar graph for GPU
gpu_bars = ax.bar(gpu_data['Cores'] - bar_width/2, gpu_data['Task2 Execution Time'], bar_width, label='GPU')

# Bar graph for CPU
cpu_bars = ax.bar(cpu_data['Cores'] + bar_width/2, cpu_data['Task2 Execution Time'], bar_width, label='CPU')

ax.set_xlabel('Number of Cores')
ax.set_ylabel('Task2 Execution Time')
ax.set_title('Comparison of Task2 Execution Time (CPU vs GPU)')
ax.set_xticks(df['Cores'])
ax.legend()

plt.show()



##################2nd plot###########################

# Separate data for GPU and non-GPU configurations
gpu_data = df[df['GPU'] == True]
cpu_data = df[df['GPU'] == False]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
opacity = 0.8

# Bar graph for GPU
gpu_bars = ax.bar(gpu_data['Cores'] - bar_width/2, gpu_data['Task1 Execution Time'], bar_width, label='GPU')

# Bar graph for CPU
cpu_bars = ax.bar(cpu_data['Cores'] + bar_width/2, cpu_data['Task1 Execution Time'], bar_width, label='CPU')

ax.set_xlabel('Number of Cores')
ax.set_ylabel('Task1 Execution Time')
ax.set_title('Comparison of Task1(LSTM) Execution Time (CPU vs GPU)')
ax.set_xticks(df['Cores'])
ax.legend()

plt.show()

#######################3RD PLOT


df_no_gpu = df[df['GPU'] == False]
df = df_no_gpu


# Plotting the line graph for 'CPU Percent' against 'Cores'
plt.figure(figsize=(10, 5))
plt.plot(df['Cores'], df['CPU Percent'], marker='o', linestyle='-', color='blue')

# Adding title and labels
plt.title('CPU Percent by Cores (GPU is False)')
plt.xlabel('Cores')
plt.ylabel('CPU Percent')

# Showing the plot
plt.grid(True)
plt.show()
















##########4rth plot######################################################
#file_path = r"C:\Users\NithinNandelli\Downloads\results.csv"
#df = pd.read_csv(file_path)
#df_no_gpu = df
print(df)
#df_no_gpu = df[df['GPU'] == 'FALSE']
df_no_gpu = df[df['GPU'] == False]






# We directly plot 'Sequential' and 'Multi-proc' against 'Cores' as we have already filtered out GPU='TRUE'
plt.figure(figsize=(10, 5))

# Plotting 'Sequential' execution time
plt.plot(df_no_gpu['Cores'], df_no_gpu['Sequential Execution time'], marker='o', label='Sequential ')

# Plotting 'Multi-proc' execution time
plt.plot(df_no_gpu['Cores'], df_no_gpu['Multi-process Execution time'], marker='s', label='parallel')

plt.title('Sequential vs parallel')
plt.xlabel('Cores')
plt.ylabel('Time in Seconds')
plt.legend()
plt.grid(True)
plt.show()







