import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pandas as pd
from scipy.linalg import solve_continuous_are, solve_discrete_are
import os


np.set_printoptions(precision=6, suppress=True)

def parse_history_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                values = line.strip().split(',')
                values = [float(value) for value in values]
            except Exception as e:
                print(f"Error parsing line: {line}")
                continue
            entry = {
                'time': values[0],
                'angX': values[1],
                'angY': values[2],
                'angZ': values[3],
                'gyroX': values[4],
                'gyroY': values[5],
                'gyroZ': values[6],
                'accX': values[7],
                'accY': values[8],
                'accZ': values[9],
                'motor1_shaft_angle': values[10],
                'motor2_shaft_angle': values[11],
                'motor1_shaft_velocity': values[12],
                'motor2_shaft_velocity': values[13],
                'thigh_Left': values[14],
                'thigh_Right': values[15],
                'ctrl0': values[16],
                'ctrl1': values[17]
            }
            data.append(entry)
    return data

def convert_to_csv(data, output_file):
    # data struct: [{'time': 0.0, 'angX': 0.0, 'angY': 0.0, 'angZ': 0.0, 'gyroX': 0.0, 'gyroY': 0.0, 'gyroZ': 0.0, 'accX': 0.0, 'accY': 0.0, 'accZ': 0.0, 'motor1_shaft_angle': 0.0, 'motor2_shaft_angle': 0.0, 'motor1_shaft_velocity': 0.0, 'motor2_shaft_velocity': 0.0, 'thigh_Left': 0.0, 'thigh_Right': 0.0, 'ctrl0': 0.0, 'ctrl1': 0.0}]
    with open(output_file, 'w') as file:
        # 写入CSV文件的标题行
        file.write('time,angX,angY,angZ,gyroX,gyroY,gyroZ,accX,accY,accZ,motor1_shaft_angle,motor2_shaft_angle,motor1_shaft_velocity,motor2_shaft_velocity,thigh_Left,thigh_Right,ctrl0,ctrl1\n')
        for entry in data:
            # 将每个条目的数据转换为CSV格式并写入文件
            file.write(f"{entry['time']},{entry['angX']},{entry['angY']},{entry['angZ']},{entry['gyroX']},{entry['gyroY']},{entry['gyroZ']},{entry['accX']},{entry['accY']},{entry['accZ']},{entry['motor1_shaft_angle']},{entry['motor2_shaft_angle']},{entry['motor1_shaft_velocity']},{entry['motor2_shaft_velocity']},{entry['thigh_Left']},{entry['thigh_Right']},{entry['ctrl0']},{entry['ctrl1']}\n")

def convert_to_array(data):
    # data struct: [{'time': 0.0, 'angX': 0.0, 'angY': 0.0, 'angZ': 0.0, 'gyroX': 0.0, 'gyroY': 0.0, 'gyroZ': 0.0, 'accX': 0.0, 'accY': 0.0, 'accZ': 0.0, 'motor1_shaft_angle': 0.0, 'motor2_shaft_angle': 0.0, 'motor1_shaft_velocity': 0.0, 'motor2_shaft_velocity': 0.0, 'thigh_Left': 0.0, 'thigh_Right': 0.0, 'ctrl0': 0.0, 'ctrl1': 0.0}]
    time = [entry['time'] for entry in data]
    angX = [entry['angX'] for entry in data]
    angY = [entry['angY'] for entry in data]
    angZ = [entry['angZ'] for entry in data]
    gyroX = [entry['gyroX'] for entry in data]
    gyroY = [entry['gyroY'] for entry in data]
    gyroZ = [entry['gyroZ'] for entry in data]
    accX = [entry['accX'] for entry in data]
    accY = [entry['accY'] for entry in data]
    accZ = [entry['accZ'] for entry in data]
    motor1_shaft_angle = [entry['motor1_shaft_angle'] for entry in data]
    motor2_shaft_angle = [entry['motor2_shaft_angle'] for entry in data]
    motor1_shaft_velocity = [entry['motor1_shaft_velocity'] for entry in data]
    motor2_shaft_velocity = [entry['motor2_shaft_velocity'] for entry in data]
    thigh_Left = [entry['thigh_Left'] for entry in data]
    thigh_Right = [entry['thigh_Right'] for entry in data]
    ctrl0 = [entry['ctrl0'] for entry in data]
    ctrl1 = [entry['ctrl1'] for entry in data]
    # combine all the data into a single array
    alldata = np.array([time, angX, angY, angZ, gyroX, gyroY, gyroZ, accX, accY, accZ, motor1_shaft_angle, motor2_shaft_angle, motor1_shaft_velocity, motor2_shaft_velocity, thigh_Left, thigh_Right, ctrl0, ctrl1])
    return alldata.T


# visualize the data: angY, gyroY
def visualize_data(data, col1, col2, name1, name2, show_points=False):
    data1 = data[:, col1]
    data2 = data[:, col2]
    # Calculate mean and standard deviation
    data1_mean = np.mean(data1)
    data1_std = np.std(data1)
    data2_mean = np.mean(data2)
    data2_std = np.std(data2)
    
    # Calculate min and max
    data1_min = np.min(data1)
    data1_max = np.max(data1)
    data2_min = np.min(data2)
    data2_max = np.max(data2)
    
    # Show data1 and data2 in two subplots
    fig, axs = plt.subplots(2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f'{name1} and {name2}')
    
    # Plot data1
    axs[0].plot(data[:, 0], data1, label=name1)
    if show_points:
        axs[0].plot(data[:, 0], data1, 'ro', markersize=2)
    axs[0].axhline(y=data1_mean, color='r', linestyle='--', label='Mean')
    axs[0].fill_between(data[:, 0], data1_mean - data1_std, data1_mean + data1_std, color='r', alpha=0.2, label='Std Dev')
    axs[0].axhline(y=data1_min, color='g', linestyle='--', label='Min')
    axs[0].axhline(y=data1_max, color='b', linestyle='--', label='Max')
    axs[0].set_title(f'{name1} (Mean: {data1_mean:.4f}, Std Dev: {data1_std:.4f}, Min: {data1_min:.4f}, Max: {data1_max:.4f})')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel(name1)
    axs[0].legend()
    
    # Plot data2
    axs[1].plot(data[:, 0], data2, label=name2)
    if show_points:
        axs[1].plot(data[:, 0], data2, 'ro', markersize=2)
    axs[1].axhline(y=data2_mean, color='r', linestyle='--', label='Mean')
    axs[1].fill_between(data[:, 0], data2_mean - data2_std, data2_mean + data2_std, color='r', alpha=0.2, label='Std Dev')
    axs[1].axhline(y=data2_min, color='g', linestyle='--', label='Min')
    axs[1].axhline(y=data2_max, color='b', linestyle='--', label='Max')
    axs[1].set_title(f'{name2} (Mean: {data2_mean:.4f}, Std Dev: {data2_std:.4f}, Min: {data2_min:.4f}, Max: {data2_max:.4f})')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel(name2)
    axs[1].legend()
    
    plt.show()

# visualize the data: angY, gyroY, ctrl0, ctrl1
def visualize_angY_gyroY_ctrl(data):
    #show angY, gyroY, ctrl0, ctrl1 in 3 subplots with space between them
    fig, axs = plt.subplots(5, figsize=(10, 8), constrained_layout=True)
    fig.suptitle('angY, gyroY, ctrl0, ctrl1')
    plot_angY = lambda ax, data, zeroy, title, ylabel: (
        ax.plot(data[0], data[1], linestyle='-', color='red'),
        ax.plot(data[0], data[1], marker='o', markersize=1, linestyle='None', color='b'),
        ax.axhline(y=zeroy, color='r', linestyle='--'),
        ax.set_title(title),
        ax.set_xlabel('time'),
        ax.set_ylabel(ylabel)
    )
    x = np.arange(0, data.shape[0], 1)
    plot_angY(axs[0], (x, data[:, 2]), 5, 'angY', 'angY')
    plot_angY(axs[1], (x, data[:, 4]), 0, 'gyroY', 'gyroY')
    plot_angY(axs[2], (x, data[:, 16]), 0, 'ctrl0', 'ctrl0')
    axs[2].twinx()
    plot_angY(axs[2], (x, data[:, 17]), 0, 'ctrl1', 'ctrl1')
    # plot wheel angle
    motor1_shaft_angle = data[:, 10]
    motor2_shaft_angle = data[:, 11]
    motor1_shaft_mean = np.mean(motor1_shaft_angle)
    motor2_shaft_mean = np.mean(motor2_shaft_angle)
    plot_angY(axs[3], (x, motor1_shaft_angle), motor1_shaft_mean, 'motor1_shaft_angle', 'motor1_shaft_angle')
    plot_angY(axs[4], (x, motor2_shaft_angle), motor2_shaft_mean, 'motor2_shaft_angle', 'motor2_shaft_angle')
    plt.show()

def testdiff(alldata):
    alldata_diff = np.diff(alldata, axis=0)
    print(alldata_diff.shape, alldata[1:, 0].shape)
    # append alldata[:, 0] to the ahead of alldata_diff
    alldata_diff = np.insert(alldata_diff, 0, alldata[1:, 0], axis=1)

    costdiff = alldata_diff[:, 1]
    ratio4ms = np.sum(costdiff >= 4) / costdiff.shape[0]
    ratio3ms = np.sum(costdiff == 3) / costdiff.shape[0]
    ratio2ms = np.sum(costdiff == 2) / costdiff.shape[0]
    print(">=4ms ratio: {:.2f}%, \n3ms ratio: {:.2f}% \n2ms ratio: {:.2f}%".format(ratio4ms*100, ratio3ms*100, ratio2ms*100))

    visualize_data(alldata_diff, 1, 3, 'time_diff', 'pitch_diff')

# visualize the data: multiple columns
def visualize_datapro(data, cols, names=None, show_points=False):
    if names is None:
        names = [f'Column {col}' for col in cols]
    
    num_cols = len(cols)
    fig, axs = plt.subplots(num_cols, figsize=(10, 4 * num_cols), constrained_layout=True)
    fig.suptitle('Data Visualization')
    
    if num_cols == 1:
        axs = [axs]
    
    for i, col in enumerate(cols):
        data_col = data[:, col]
        data_mean = np.mean(data_col)
        data_std = np.std(data_col)
        data_min = np.min(data_col)
        data_max = np.max(data_col)
        
        axs[i].plot(data[:, 0], data_col, label=names[i])
        if show_points:
            axs[i].plot(data[:, 0], data_col, 'ro', markersize=2)
        axs[i].axhline(y=data_mean, color='r', linestyle='--', label='Mean')
        axs[i].fill_between(data[:, 0], data_mean - data_std, data_mean + data_std, color='r', alpha=0.2, label='Std Dev')
        axs[i].axhline(y=data_min, color='g', linestyle='--', label='Min')
        axs[i].axhline(y=data_max, color='b', linestyle='--', label='Max')
        axs[i].set_title(f'{names[i]} (Mean: {data_mean:.4f}, Std Dev: {data_std:.4f}, Min: {data_min:.4f}, Max: {data_max:.4f})')
        axs[i].set_xlabel('time')
        axs[i].set_ylabel(names[i])
        axs[i].legend()
    # save the figure
    plt.savefig('data_visualization.png')
    # plt.show()

def visualize_datapro2(data, cols, names=None, show_points=False, combine_indices=None, show_stats=True, subnum=3):
    if names is None:
        names = [f'Column {col}' for col in cols]
    
    if combine_indices is None:
        combine_indices = []
    
    # 计算子图数量
    num_subplots = len(combine_indices)
    
    # 计算行数和列数
    num_rows = (num_subplots + subnum - 1) // subnum
    num_cols = min(subnum, num_subplots)
    print(num_rows, num_cols)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 4 * num_rows), constrained_layout=True)
    fig.suptitle('Data Visualization')
    
    if num_rows == 1:
        axs = [[axs]]
    elif num_cols == 1:
        axs = [[ax] for ax in axs]
    
    for subplot_index, indices in enumerate(combine_indices):
        row = subplot_index // num_cols
        col_index = subplot_index % num_cols
        print(row, col_index)
        ax = axs[row][col_index]
        
        # 处理左Y轴数据
        print(indices[0])
        if isinstance(indices[0], list):
            left_indices = indices[0]
            right_indices = indices[1]
            
            # 绘制左Y轴数据
            for idx in left_indices:
                data_col = data[:, cols[idx]]
                ax.plot(data[:, 0], data_col, label=names[idx])
                if show_points:
                    ax.plot(data[:, 0], data_col, 'ro', markersize=2)
                if show_stats:
                    data_mean = np.mean(data_col)
                    ax.axhline(y=data_mean, color='r', linestyle='--', label=f'{names[idx]} Mean: {data_mean:.4f}')
                    data_std = np.std(data_col)
                    ax.fill_between(data[:, 0], data_mean - data_std, data_mean + data_std, color='r', alpha=0.2, label=f'{names[idx]} Std Dev: {data_std:.4f}')
            
            # 创建右Y轴并绘制数据
            ax_right = ax.twinx()
            # draw zero line 
            ax_right.axhline(y=0, color='k', linestyle=':', label='Zero Line')
            for idx in right_indices:
                data_col = data[:, cols[idx]]
                print("names[idx]: ", names[idx])
                ax_right.plot(data[:, 0], data_col, '--', label=names[idx])
                if show_points:
                    ax_right.plot(data[:, 0], data_col, 'bo', markersize=2)
                if show_stats:
                    data_mean = np.mean(data_col)
                    ax_right.axhline(y=data_mean, color='b', linestyle=':', label=f'{names[idx]} Mean: {data_mean:.4f}')
                    data_std = np.std(data_col)
                    ax_right.fill_between(data[:, 0], data_mean - data_std, data_mean + data_std, color='b', alpha=0.2, label=f'{names[idx]} Std Dev: {data_std:.4f}')
            
            # 合并图例
            lines, labels = ax.get_legend_handles_labels()
            lines_right, labels_right = ax_right.get_legend_handles_labels()
            print(lines, labels, lines_right, labels_right)
            ax.legend(lines + lines_right, labels + labels_right)
        else:
            # 原有简单组合逻辑
            for idx in indices:
                data_col = data[:, cols[idx]]
                ax.plot(data[:, 0], data_col, label=names[idx])
                if show_points:
                    ax.plot(data[:, 0], data_col, 'ro', markersize=2)
                if show_stats:
                    data_mean = np.mean(data_col)
                    ax.axhline(y=data_mean, color='r', linestyle='--', label=f'{names[idx]} Mean: {data_mean:.4f}')
                    data_std = np.std(data_col)
                    ax.fill_between(data[:, 0], data_mean - data_std, data_mean + data_std, color='r', alpha=0.2, label=f'{names[idx]} Std Dev: {data_std:.4f}')

            # 添加图例
            ax.legend()
        
        ax.grid(True)
        ax.set_xlabel('time')
        ax.set_ylabel('Values')
    
    plt.savefig('data_visualization.png')
    plt.show()


def mapping_thighs_velocity(thigh_left_vel, thigh_right_vel):
    # 0~360deg vs 0~4096, there is dispersion
    # in simbot, from squat to stand is negative speed
    # in realbot, left, from squat to stand is positive speed, right, from squat to stand is negative speed
    scale = 2 * 3.1415 / 4096
    sim_thigh_left_vel = -thigh_left_vel * scale
    sim_thigh_right_vel = thigh_right_vel * scale
    
    return sim_thigh_left_vel, sim_thigh_right_vel

def mapping_thighs_position(thigh_left, thigh_right):
    # 0~360deg vs 0~4096, there is dispersion
    thigh_left_squad = 2136
    thigh_left_stand = 2676
    thigh_right_squad = 2019
    thigh_right_stand = 1479
    sim_thigh_left_squad = 0  # 0.439; //0.388; //0.2869;
    sim_thigh_left_stand = -0.5  # -0.26599;
    sim_thigh_right_squad = 0  # 0.388; //0.2869;
    sim_thigh_right_stand = -0.5

    # (y-sim_left_squad)/(x-thigh_left_squad)=(sim_left_stand-sim_left_squad)/(thigh_left_stand-thigh_left_squad)
    sim_thigh_left = sim_thigh_left_squad + (sim_thigh_left_stand - sim_thigh_left_squad) * (thigh_left - thigh_left_squad) / (thigh_left_stand - thigh_left_squad)
    # (y-sim_right_stand)/(x-thigh_right_stand)=(sim_right_squad-sim_right_stand)/(thigh_right_squad-thigh_right_stand)
    sim_thigh_right = sim_thigh_right_stand + (sim_thigh_right_squad - sim_thigh_right_stand) * (thigh_right - thigh_right_stand) / (thigh_right_squad - thigh_right_stand)

    return sim_thigh_left, sim_thigh_right

def parse_history_file2(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                values = line.strip().split(',')
                values = [float(value) for value in values]
            except Exception as e:
                print(f"Error parsing line: {line}")
                continue
            ang_gyros = np.radians(values[1:7])
            thigh_left, thigh_right = mapping_thighs_position(values[14], values[15])
            thigh_left_ctrl, thigh_right_ctrl = mapping_thighs_position(values[18], values[19])
            thigh_left_vel, thigh_right_vel = mapping_thighs_velocity(values[16], values[17])
            entry = {
                'time': values[0],
                'angX': ang_gyros[0],
                'angY': ang_gyros[1],
                'angZ': ang_gyros[2],
                'gyroX': ang_gyros[3],
                'gyroY': ang_gyros[4],
                'gyroZ': ang_gyros[5],
                'accX': values[7],
                'accY': values[8],
                'accZ': values[9],
                'motor1_shaft_angle': values[10],
                'motor2_shaft_angle': values[11],
                'motor1_shaft_velocity': values[12],
                'motor2_shaft_velocity': values[13],
                'thigh_Left': thigh_left,
                'thigh_Right': thigh_right,
                'thigh_left_velocity': thigh_left_vel,
                'thigh_right_velocity': thigh_right_vel,
                'thigh_left_control': thigh_left_ctrl,
                'thigh_right_control': thigh_right_ctrl,
                'wheel_left_control': values[20],
                'wheel_right_control': values[21]
            }
            data.append(entry)
    return data


def convert_to_array2(data):
    # data struct: [{'time': 0.0, 'angX': 0.0, 'angY': 0.0, 'angZ': 0.0, 'gyroX': 0.0, 'gyroY': 0.0, 'gyroZ': 0.0, 'accX': 0.0, 'accY': 0.0, 'accZ': 0.0, 'motor1_shaft_angle': 0.0, 'motor2_shaft_angle': 0.0, 'motor1_shaft_velocity': 0.0, 'motor2_shaft_velocity': 0.0, 'thigh_Left': 0.0, 'thigh_Right': 0.0, 'ctrl0': 0.0, 'ctrl1': 0.0}]
    time = [entry['time'] for entry in data]
    angX = [entry['angX'] for entry in data]
    angY = [entry['angY'] for entry in data]
    angZ = [entry['angZ'] for entry in data]
    gyroX = [entry['gyroX'] for entry in data]
    gyroY = [entry['gyroY'] for entry in data]
    gyroZ = [entry['gyroZ'] for entry in data]
    accX = [entry['accX'] for entry in data]
    accY = [entry['accY'] for entry in data]
    accZ = [entry['accZ'] for entry in data]
    motor1_shaft_angle = [entry['motor1_shaft_angle'] for entry in data]
    motor2_shaft_angle = [entry['motor2_shaft_angle'] for entry in data]
    motor1_shaft_velocity = [entry['motor1_shaft_velocity'] for entry in data]
    motor2_shaft_velocity = [entry['motor2_shaft_velocity'] for entry in data]
    thigh_Left = [entry['thigh_Left'] for entry in data]
    thigh_Right = [entry['thigh_Right'] for entry in data]
    thigh_Left_velocity = [entry['thigh_left_velocity'] for entry in data]
    thigh_Right_velocity = [entry['thigh_right_velocity'] for entry in data]
    thigh_Left_control = [entry['thigh_left_control'] for entry in data]
    thigh_Right_control = [entry['thigh_right_control'] for entry in data]
    wheel_Left_control = [entry['wheel_left_control'] for entry in data]
    wheel_Right_control = [entry['wheel_right_control'] for entry in data]
    # combine all the data into a single array
    alldata = np.array([time, angX, angY, angZ, gyroX, gyroY, gyroZ, 
                        accX, accY, accZ, motor1_shaft_angle, motor2_shaft_angle, motor1_shaft_velocity, motor2_shaft_velocity, 
                        thigh_Left, thigh_Right, thigh_Left_velocity, thigh_Right_velocity, 
                        thigh_Left_control, thigh_Right_control, wheel_Left_control, wheel_Right_control])
    return alldata.T

def detect_high_frequency_vibration(data, threshold):
    # Perform FFT
    freq_domain = fft(data)
    magnitudes = np.abs(freq_domain)

    # Check if any frequency magnitude exceeds the threshold
    return np.any(magnitudes > threshold)

def remove_high_frequency_vibration(data, alpha):
    # Design an exponential moving average filter
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    return filtered_data

def process_realtime_data(data_stream, threshold, alpha, sample_rate):
    buffer_size = int(sample_rate / (1 - alpha)) * 2  # Buffer size for FFT and filtering
    data_buffer = []
    
    for new_data in data_stream:
        # Add new data to buffer
        data_buffer.append(new_data)
        
        # Ensure buffer does not exceed the buffer size
        if len(data_buffer) > buffer_size:
            data_buffer.pop(0)
        
        # Detect high-frequency vibration
        if detect_high_frequency_vibration(data_buffer, threshold):
            # Remove high-frequency vibration
            filtered_data = remove_high_frequency_vibration(np.array(data_buffer), alpha)
            yield filtered_data[-1]
        else:
            yield new_data

def plot_frequency_signal(data, indices, sample_rate, num_points=None, labels=None, title='Frequency Signal', xlabel='Frequency (Hz)', ylabel='Magnitude'):
    plt.figure(figsize=(10, 6))
    
    for i, col in enumerate(indices):
        # Perform FFT
        freq_domain = fft(data[:, col])
        magnitudes = np.abs(freq_domain)
        frequencies = np.fft.fftfreq(len(data[:, col]), d=1/sample_rate)

        # Limit the number of points to display if specified
        if num_points is not None:
            magnitudes = magnitudes[:num_points]
            frequencies = frequencies[:num_points]

        # Plot frequency signal
        label = labels[i] if labels is not None else f'Signal {col}'
        plt.plot(frequencies[:len(frequencies)//2], magnitudes[:len(magnitudes)//2], label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig('frequency_signal.png')
    plt.show()

def analyze_frequency_signal(alldata, showfreq=False):
    # Detect and remove high-frequency vibration from sensor data
    vibration_threshold = 1.0  # Example threshold value
    alpha = 0.5  # Example alpha value for exponential moving average filter
    sample_rate = 100.0  # Example sample rate in Hz

    if showfreq:
        indices = [5, 6, 20, 21]
        labels = ['gyroY' ,'gyroZ', 'wheel_left_control', 'wheel_right_control']
        indices = [4, 15, 19, 18]
        labels = ['gyroX', 'thigh_Right', 'thigh_right_control', 'thigh_left_control']
        plot_frequency_signal(alldata, indices, sample_rate, num_points=None, labels=labels, title='Frequency Signal for Sensors', xlabel='Frequency (Hz)', ylabel='Magnitude')

    for i in range(1, alldata.shape[1]):  # Skip the time column
        data_stream = alldata[:, i]
        processed_data = list(process_realtime_data(data_stream, vibration_threshold, alpha, sample_rate))
        alldata[:, i] = processed_data

def main():
    dstart = 0
    dlen = 200
    file_path = 'data/data77_use.txt'  # 替换为你的文件路径
    file_path = 'data/history - 2025-01-01T173302.190.txt'
    file_path = 'stuff/20250211_2/history_-0.4_3.txt'
    """
    // sensors: [time:0, angX:1, angY:2, angZ:3, gyroX:4, gyroY:5, gyroZ:6, accX:7, accY:8, accZ:9, 
    //           motor1_shaft_angle:10, motor2_shaft_angle:11, motor1_shaft_velocity:12, motor2_shaft_velocity:13, 
    //           thigh_Left:14, thigh_Right:15, thigh_Left_velocity:16, thigh_Right_velocity:17]
    // control: [thigh_left_control:0, thigh_right_control:1, wheel_left_control:2, wheel_right_control:3]
    """
    file_path = 'data/history - 2025-01-28T053756.456.mark.txt'
    file_path = 'data/history - 2025-01-28T215041.516.mark.txt'
    file_path = 'data/history - 2025-01-28T223352.290.mark.txt'
    file_path = 'stuff/history - 2025-02-06T172039.466.high.checkroll.txt'
    file_path = 'stuff/history - 2025-02-06T195424.346.high.checkroll2.txt'
    file_path = 'stuff/history - 2025-02-06T201617.647.wengweng.txt'

    file_path = 'stuff/20250211_2/history_-0.4_3.txt'
    file_path = 'stuff/20250211_2/history_-0.4_2.txt'
    file_path = 'stuff/20250211_2/history_-0.4_1.txt'
    file_path = 'stuff/20250211_2/history_-0.4_5.txt'
    # file_path = "stuff/history (56).txt"
    # file_path = "stuff/history (60).txt"
    file_path = "stuff/history (62).txt"
    file_path = "stuff/history - 2025-07-16T174038.388.txt"
    file_path = "stuff/history - 2025-07-16T180158.846.txt"
    file_path = "stuff/history - 2025-07-17T163505.483.txt"
    parsed_data = parse_history_file2(file_path)
    parsed_data = parsed_data[dstart:dstart+dlen]
    
    alldata = convert_to_array2(parsed_data)
    alldata[:, 0] = alldata[:, 0] - alldata[0, 0]
    print(alldata.shape)
    # print(alldata[0])
    # print(alldata[-1])
    timelen = alldata[-1, 0] - alldata[0, 0]
    print("timespan:", timelen)
    print("time per step:", timelen/dlen)

    if True:
        analyze_frequency_signal(alldata)
    # visualize_data(alldata, 1, -3, 'roll', 'thigh_ctrl2', True)
    # visualize_angY_gyroY_ctrl(alldata)

    # visualize_datapro2(alldata, [1, 4, 15, 17, 19, 18], 
    #                   ['angX', 'gyroX', 'thigh_Right' ,'thigh_Right_velocity', 'thigh_right_control', 'thigh_left_control'], 
    #                   show_points=False, combine_indices=[[2, 4, 5]], show_stats=False, subnum=2)
    # visualize_datapro2(alldata, [2, 3, 5, 6, 12, 13, 20, 21], 
    #                   ['angY', 'angZ', 'gyroY' ,'gyroZ', 'motor1_shaft_velocity', 'motor2_shaft_velocity', 'wheel_left_control', 'wheel_right_control'], 
    #                   show_points=False, combine_indices=[[0, 1], [2, 3], [4, 5], [6, 7]], show_stats=False, subnum=2)
    # visualize_datapro2(alldata, [2, 1, 5, 4, 12, 13, 20, 21], 
    #                   ['angY', 'angX', 'gyroY' ,'gyroX', 'motor1_shaft_velocity', 'motor2_shaft_velocity', 'wheel_left_control', 'wheel_right_control'], 
    #                   show_points=False, combine_indices=[[[0, 1], [6, 7]], [[4, 5], [6, 7]]], show_stats=False, subnum=1)
    index = [10, 12, 20]
    names = ['motor1_shaft_angle', 'motor1_shaft_velocity', 'wheel_left_control']
    combine_indices = [[[0, 1], [2]]]
    visualize_datapro2(alldata, index, names, show_points=False, combine_indices=combine_indices, show_stats=False, subnum=1)



def lqr_with_realbotdata_deltax(data_file):
    # data_file = 'stuff/history - 2025-02-06T195424.346.high.checkroll2.txt'
    # 读取数据文件
    data = pd.read_csv(data_file, header=None, 
                    names=['time', 'angX', 'angY', 'angZ', 'gyroX', 'gyroY', 'gyroZ', 
                            'accX', 'accY', 'accZ', 'motor1_shaft_angle', 'motor2_shaft_angle',
                            'motor1_shaft_velocity', 'motor2_shaft_velocity', 'thigh_Left',
                            'thigh_Right', 'thigh_Left_velocity', 'thigh_Right_velocity',
                            'thigh_left_ctrl', 'thigh_right_ctrl', 'wheel_left_ctrl', 
                            'wheel_right_ctrl'])
    # time
    time = data['time'].values
    delta_time = (time[1:] - time[:-1]) / 1000 # 时间间隔（单位：秒）
    # 提取状态量和控制量
    states = data[['angX', 'gyroX']].values  # 状态量：angX 和 gyroX
    # convert to radian
    states = np.radians(states)
    print(states.shape)
    controls = data[['thigh_left_ctrl', 'thigh_right_ctrl']].values  # 控制量
    # convert thigh control with mapping_thighs_position
    for i in range(controls.shape[0]):
        thigh_left, thigh_right = mapping_thighs_position(controls[i, 0], controls[i, 1])
        controls[i, 0] = thigh_left
        controls[i, 1] = thigh_right
    controls = controls[:-1]

    # 计算状态变化（当前状态与下一个状态的差）
    delta_states = (states[1:] - states[:-1]) / delta_time[:, np.newaxis]
    states_current = states[:-1]
    print("delta_states[:10]:", delta_states[:10])

    # 构建输入输出数据矩阵
    n_states = states.shape[1]  # 状态维度：2
    n_controls = controls.shape[1]  # 控制量维度：2

    A_matrix = np.zeros((n_states, n_states))
    B_matrix = np.zeros((n_states, n_controls))

    # 迭代计算系统矩阵 A 和 B
    for i in range(len(states_current)):
        delta_state = delta_states[i]
        state = states_current[i]
        control = controls[i]
        
        # 估计状态变化率 (简化为线性关系)
        A_matrix += np.outer(delta_state, state) / len(states_current)
        B_matrix += np.outer(delta_state, control) / len(states_current)
    # calc fitting error
    error = np.mean(np.abs(delta_states - np.dot(A_matrix, states_current.T).T - np.dot(B_matrix, controls.T).T), axis=0)
    print(f"fitting error: {error}")
    # 定义LQR权重矩阵
    Q = np.diag([10.0, 1.0])  # 状态权重：angX更重要，gyroX次要
    R = np.diag([1.0, 1.0])  # 控制输入权重

    # 计算Riccati方程的解P
    P = solve_continuous_are(A_matrix, B_matrix, Q, R)

    # 计算最优反馈增益矩阵K
    K = -np.dot(np.linalg.inv(R), np.dot(B_matrix.T, P))

    print("优化后的LQR控制增益矩阵K:")
    print(K)

def getfiles(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            files.append(os.path.join(folder, file))
    # sort the files by time in name
    files.sort(key=lambda x: x.split(' - ')[1].split('.')[0])
    return files

def getfiles2(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            files.append(os.path.join(folder, file))
    # filename format: path/history_val_num.txt, extract val and sort by val
    # get val
    numfunc = lambda x: int(x.split('_')[-1].split('.')[0])
    valfunc = lambda x: float(x.split('_')[-2])
    files.sort(key=valfunc)
    vals = [valfunc(file) for file in files]
    return files, vals

def lqr_with_realbotdata(data_file, val=0, discrete=False):
    # data_file = 'stuff/history - 2025-02-06T195424.346.high.checkroll2.txt'
    # data_file = 'stuff/history - 2025-02-06T172039.466.high.checkroll.txt'
    # get folder named stuff/20250211 files

    
    # 读取数据文件
    data = pd.read_csv(data_file, header=None,
                       names=['time', 'angX', 'angY', 'angZ', 'gyroX', 'gyroY', 'gyroZ', 
                            'accX', 'accY', 'accZ', 'motor1_shaft_angle', 'motor2_shaft_angle',
                            'motor1_shaft_velocity', 'motor2_shaft_velocity', 'thigh_Left',
                            'thigh_Right', 'thigh_Left_velocity', 'thigh_Right_velocity',
                            'thigh_left_ctrl', 'thigh_right_ctrl', 'wheel_left_ctrl', 
                            'wheel_right_ctrl'])
    
    A, B, K = lqr_data(data, discrete)

def lqr_data(data, discrete=False, minus_mean_ctrl=False):
    max_len = 50000
    # 提取状态量和控制量
    time = data['time'].values[:max_len]
    # print("time[:5]:\n", time[:5])
    states = data[['angX', 'gyroX']].values[:max_len]  # 状态量：angX 和 gyroX
    # print("states[:5]:\n", states[:5])
    states = np.radians(states)  # 转换为弧度
    # print("states.shape:", states.shape)
    controls = data[['thigh_left_ctrl', 'thigh_right_ctrl']].values[:max_len]  # 控制量
    # print("controls[:5]:\n", controls[:5])
    if minus_mean_ctrl:
        print('mean_ctrl.shape:', data['mean_ctrl'].values[:max_len].shape)
        mean_ctrl = data['mean_ctrl'].values[:max_len]
    
    # 将控制量映射到物理位置
    for i in range(controls.shape[0]):
        thigh_left, thigh_right = mapping_thighs_position(
            controls[i, 0], controls[i, 1])
        controls[i, 0] = thigh_left
        controls[i, 1] = thigh_right
    mean_controls = np.mean(controls, axis=0)
    print(f"Mean controls: {mean_controls}")
    # mean_controls = np.array([val, val])
    print(f"Original controls: {mean_controls}")
    
    
    # 计算状态变化（当前状态与下一个状态的差）
    delta_states = states[1:] - states[:-1]
    states_current = states[:-1]

    x = states
    u = controls
    print('mean_ctrl[:10]:', mean_ctrl[:10])
    print("u[:10]:\n", u[:10])
    
    # 构建最小二乘法输入矩阵
    n_states = x.shape[1]  # 状态量的数量
    n_controls = u.shape[1]     # 控制量的数量
    
    # 参数初始化
    n_samples = len(x) - 1  # 样本数
    n_state = x[0].shape[0]
    n_input = u[0].shape[0]

    X = np.zeros((n_samples, n_state + n_input))
    y = np.zeros((n_samples, n_state))

    for k in range(n_samples):
        X[k, :n_state] = x[k]
        if minus_mean_ctrl:
            X[k, n_state:] = u[k] - mean_ctrl[k]
        else:
            X[k, n_state:] = u[k]
        
        if discrete:
            y[k] = x[k+1]
        else:
            delta_x = x[k+1] - x[k]
            dt = (time[k+1] - time[k])/1000  # 时间间隔（单位：秒）
            y[k] = delta_x / dt
    print("y[:10]:\n", y[:10])
    print("X[:10]:\n", X[:10])
    X_mean = np.mean(X, axis=0)
    print("X_mean:", X_mean)
    print("y[:, 0]_min: {}, y[:, 0]_max: {}".format(np.min(y[:, 0]), np.max(y[:, 0])))
    print("abs(y_mean):", np.mean(np.abs(y), axis=0))
    # 使用最小二乘法求解系统矩阵 [A; B]
    theta, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    print(f"Estimated theta:\n{theta.shape}, X.shape: {X.shape}, y.shape: {y.shape}")
    # print residuals
    # error_mean = residuals / n_samples
    # print (f"error_mean: {error_mean}")
    # calc error by mean(abs(y - X*theta), axis=0)
    error = np.mean(np.abs(y - X @ theta), axis=0)
    print(f"fitting error: {error}")
    
    # 将 θ 分割为 A 和 B 矩阵
    
    A = theta[:n_states].reshape(n_states, n_states).T
    B = theta[n_states:].reshape(n_states, n_controls).T
    
    print(f"Estimated A matrix:\n{A}")
    print(f"Estimated B matrix:\n{B}")
    
    # 定义状态空间模型的维度
    nx = A.shape[0]
    nu = B.shape[1]
    
    # 设置权重矩阵 Q 和 R（根据具体需求调整）
    Q = np.eye(nx)  # 状态误差权重矩阵
    # Q[0, 0] = 1.0  # angX 权重更高
    # Q[1, 1] = 0.1 # gyroX 权重更低
    R = np.eye(nu)  # 控制输入权重矩阵
    
    try:
        if discrete:
            # 解离散时间代数 Riccati 方程 (DARE)
            P = solve_discrete_are(A, B, Q, R)
            
            # 计算反馈增益矩阵 K
            S = R + B.T @ P @ B
            K = -(S @ (B.T @ P @ A))
        else:
            # 解连续时间代数 Riccati 方程 (CARE)
            P = solve_continuous_are(A, B, Q, R)
            
            # 计算反馈增益矩阵 K
            K = -np.linalg.inv(R) @ (B.T @ P)
        
        print("反馈增益矩阵 K:")
        print(K)
    except Exception as e:
        print(f"求解 Riccati 方程时出错: {e}")
    
    return A, B, K

def lqr_allfiles():
    folder = 'stuff/20250211_2'
    files, vals = getfiles2(folder)
    for i, file in enumerate(files):
        val = vals[i]
        print(f"Processing file: {file}, val: {val}")
        A, B, K = lqr_with_realbotdata(file, val=val, discrete=False)
        # lqr_with_realbotdata_deltax(file)
        print()

def combine_allfiles_then_calc():
    folder = 'stuff/20250211_2'
    files, vals = getfiles2(folder)
    # combine all the data
    alldata = []
    for i, file in enumerate(files):
        val = vals[i]
        print(f"Processing file: {file}, val: {val}")
        data = pd.read_csv(file, header=None,
                       names=['time', 'angX', 'angY', 'angZ', 'gyroX', 'gyroY', 'gyroZ', 
                            'accX', 'accY', 'accZ', 'motor1_shaft_angle', 'motor2_shaft_angle',
                            'motor1_shaft_velocity', 'motor2_shaft_velocity', 'thigh_Left',
                            'thigh_Right', 'thigh_Left_velocity', 'thigh_Right_velocity',
                            'thigh_left_ctrl', 'thigh_right_ctrl', 'wheel_left_ctrl', 
                            'wheel_right_ctrl'])
        # add val to the data, its length is same to data length
        val_data = np.full((data.shape[0], 1), val)
        print("val_data.shape:", val_data.shape)
        data['mean_ctrl'] = val_data
        alldata.append(data)
    alldata = pd.concat(alldata)
    A, B, K = lqr_data(alldata, discrete=False, minus_mean_ctrl=True)


# def simulate_system(K, noise_std=0.0):
#     """仿真系统运行，返回代价函数J(K)"""
#     x = np.array([0.5, -0.3])  # 初始状态
#     T = 50  # 时间步数
#     J = 0.0
#     for _ in range(T):
#         u = np.dot(K, x) + np.random.normal(0, noise_std)  # 加噪声模拟真机扰动
#         x = np.dot(A, x) + np.dot(B, u)
#         J += np.dot(x.T, np.dot(Q, x)) + np.dot(u.T, np.dot(R, u))
#     return J / T  # 平均代价

# def IFT_optimization():
#     K = K_init.copy()
#     for i in range(max_iter):
#         # 实验1：计算J(K)
#         J_current = simulate_system(K, noise_std=0.05)
        
#         # 实验2：计算J(K + delta_K)
#         delta_K = epsilon * np.random.randn(*K.shape)  # 随机扰动方向
#         J_perturbed = simulate_system(K + delta_K, noise_std=0.05)
        
#         # 梯度估计
#         grad_J = (J_perturbed - J_current) / epsilon
        
#         # 参数更新（投影到稳定区域，可选）
#         K_new = K - alpha * grad_J * delta_K  # 注：delta_K作为扰动方向
        
#         # 限制控制增益范围（安全约束）
#         K_new = np.clip(K_new, -2.0, 2.0)
        
#         # 更新K
#         K = K_new.copy()
#         J_history.append(J_current)
#         print(f"Iter {i+1}: J = {J_current:.4f}, K = {K.flatten()}")
    
#     return K


def parse_history_file_rldata(file_path):
    data_label = ['time', 'cmd0', 'cmd1', 'cmd2', 'cmd3', 'gyroX', 'gyroY', 'gyroZ', 
                  'sinYaw', 'sinPitch', 'sinRoll', 'left_thigh_pos', 'left_wheel_sinpos', 'right_thigh_pos', 'right_wheel_sinpos',
                  'left_thigh_vel', 'left_wheel_vel', 'right_thigh_vel', 'right_wheel_vel', 'left_wheel_linvel', 'right_wheel_linvel',
                  'lasct_act0', 'lasct_act1', 'lasct_act2', 'lasct_act3']
    # data_label = ['time', 'cmd0', 'cmd1', 'cmd2', 'cmd3', 'gyroX', 'gyroY', 'gyroZ', 
    #               'sinPitch', 'sinRoll', 'left_thigh_pos', 'left_wheel_sinpos', 'right_thigh_pos', 'right_wheel_sinpos',
    #               'left_wheel_vel', 'right_wheel_vel', 'left_wheel_linvel', 'right_wheel_linvel',
    #               'lasct_act0', 'lasct_act1', 'lasct_act2', 'lasct_act3']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                values = line.strip().split(',')
                values = [float(value) for value in values]
            except Exception as e:
                print(f"Error parsing line: {line}")
                continue
            entry = {}
            for label, value in zip(data_label, values[:len(data_label)]):
                entry[label] = value
            data.append(entry)
    return data, data_label

def convert_to_array_rldata(data, labels):
    # convert data into a ndarray
    data_array = np.array([[entry[label] for label in labels] for entry in data])
    return data_array

def main_rldata():
    """
        data_label = ['time', 'cmd0', 'cmd1', 'cmd2', 'cmd3', 'gyroX', 'gyroY', 'gyroZ', 
                  'sinYaw', 'sinPitch', 'sinRoll', 'left_thigh_pos', 'left_wheel_sinpos', 'right_thigh_pos', 'right_wheel_sinpos',
                  'left_wheel_vel', 'right_thigh_vel', 'left_wheel_linvel', 'right_wheel_linvel',
                  'lasct_act0', 'lasct_act1', 'lasct_act2', 'lasct_act3']
    """
    dstart = 0 #1000
    dlen = 1000 #1100
    
    # file_path = "stuff/history - 2025-06-07T112218.107.txt"
    # file_path = "stuff/history - 2025-06-07T170151.471.txt"
    # file_path = "stuff/20250610_leftwheel/history - 2025-06-10T113912.646.txt"
    # file_path = "stuff/20250610_leftwheel/history - 2025-06-10T114452.642.txt"
    # file_path = "stuff/20250611_leftthigh/history - 2025-06-11T140444.329.txt"
    # file_path = "stuff/history - 2025-06-11T212553.089.txt"
    #file_path = "stuff/history - 2025-06-11T213605.886.txt"
    # file_path = "stuff/history - 2025-06-13T200418.478.txt"
    # file_path = "stuff/history - 2025-06-13T201702.039.txt"
    # file_path = "stuff/history - 2025-06-14T135302.843.txt"
    # file_path = "stuff/history - 2025-06-14T135322.841.txt"
    # file_path = "stuff/history - 2025-06-14T135252.844.txt"   
    # file_path = "stuff/history - 2025-06-14T154343.897.txt"
    # file_path = "stuff/history - 2025-06-14T152529.980.txt"
    # file_path = "stuff/history - 2025-06-14T153017.360.txt"
    # file_path = "stuff/history - 2025-06-14T181446.228.txt"
    # file_path = "stuff/history - 2025-06-14T212502.570.txt"
    # file_path = "stuff/history - 2025-06-14T214219.946.txt"
    # file_path = "stuff/history - 2025-06-15T213611.808.txt"
    # file_path = "stuff/history - 2025-06-18T091445.368.txt"
    # file_path = "stuff/history - 2025-06-21T170030.734.txt"
    # file_path = "stuff/history - 2025-07-01T111129.888.txt"
    # file_path = "stuff/history - 2025-07-07T102003.113.txt"
    # file_path = "stuff/history - 2025-07-07T101953.112.txt"
    file_path = "stuff/history (7).txt"
    file_path = "stuff/history (6).txt"
    parsed_data, datalabel = parse_history_file_rldata(file_path)
    parsed_data = parsed_data[dstart:dstart+dlen]
    # print("parsed_data:", parsed_data)
    
    alldata = convert_to_array_rldata(parsed_data, datalabel)
    alldata[:, 0] = alldata[:, 0] - alldata[0, 0]
    print(alldata.shape)

    # print std and mean for each column
    for i in range(alldata.shape[1]):
        print(f"{datalabel[i]} std: {np.std(alldata[:, i]):.4f}, mean: {np.mean(alldata[:, i]):.4f}")

    timelen = alldata[-1, 0] - alldata[0, 0]
    print("timespan:", timelen)
    print("time per step:", timelen/dlen)

    # tag_str = ['lasct_act0', 'lasct_act1', 'lasct_act2', 'lasct_act3', 
    #     'gyroX', 'gyroY', 'gyroZ', 'sinPitch', 'sinRoll', 
    #     'left_wheel_sinpos', 'right_wheel_sinpos', 'left_wheel_linvel', 'right_wheel_linvel',
    #     'left_thigh_pos', 'right_thigh_pos', 'left_thigh_vel', 'right_thigh_vel']
    # tag_index = [datalabel.index(tag) for tag in tag_str]
    # print("tag_index:", tag_index)
    # visualize_datapro2(alldata, tag_index, tag_str, 
    #                   show_points=False, combine_indices=[[0, 1], [2, 3], [4, 5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], 
    #                   show_stats=False, subnum=2)

    # tag_str = ['lasct_act2', 'left_wheel_sinpos', 'left_wheel_vel']
    # tag_index = [datalabel.index(tag) for tag in tag_str]
    # print("tag_index:", tag_index)
    # visualize_datapro2(alldata, tag_index, tag_str, 
    #                   show_points=False, combine_indices=[[[0], [1]], [[0], [2]]], 
    #                   show_stats=True, subnum=1)

    # tag_str = ['lasct_act0', 'left_thigh_pos', 'left_thigh_vel']
    # tag_index = [datalabel.index(tag) for tag in tag_str]
    # print("tag_index:", tag_index)
    # visualize_datapro2(alldata, tag_index, tag_str, 
    #                   show_points=False, combine_indices=[[[0], [1]], [[0], [2]]], 
    #                   show_stats=True, subnum=1)

    # tag_str = ['lasct_act0', 'lasct_act1', 'left_thigh_pos', 'right_thigh_pos', 'sinPitch', 'sinRoll', 'cmd2', 'gyroX', 'gyroY', 'gyroZ']
    # combine_indices=[[[0, 6, 2], [5]], [[0, 1, 6], [4, 5]], [7, 8, 9]]
    # tag_str = ['cmd2', 'left_thigh_pos', 'right_thigh_pos', 'sinRoll', 'gyroX']
    # combine_indices=[[[0, 1, 2], [3]], [[3], [4]]]
    tag_str = ['cmd0', 'gyroY', 'sinPitch', 'left_wheel_sinpos', 'right_wheel_sinpos', 
        'left_wheel_vel', 'right_wheel_vel', 'lasct_act2', 'lasct_act3', 'sinYaw', 
        'left_thigh_pos', 'right_thigh_pos', 'sinRoll']
    combine_indices=[[[0], [1]], [[0], [2, 12]], [[2], [3, 4]], [[2], [5, 6]], [[0], [7, 8]], [[9], [10, 11]]]
    tag_index = [datalabel.index(tag) for tag in tag_str]
    print("tag_index:", tag_index)
    visualize_datapro2(alldata, tag_index, tag_str, 
                      show_points=False, combine_indices=combine_indices, 
                      show_stats=False, subnum=2)


if __name__ == '__main__':
    # main()
    main_rldata()

    # lqr_with_realbotdata()
    # lqr_with_realbotdata(True)
    # lqr_with_realbotdata_deltax()
    # lqr_allfiles()
    # combine_allfiles_then_calc()