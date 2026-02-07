import numpy as np
import scipy.io.wavfile as wave
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

def wiener_filter(filename, clean_path, output, metrics='rmse', visuralize=0):
    '''
    filename:混合高斯噪声的音频文件路径
    clean_path:干净的音频文件路径
    output:输出路径
    metrics:选择评价指标
    '''
    sample_rate, data = wave.read(filename) #sample_rate:8000, data:numpy
    data = data - np.mean(data)
    data = data / np.max(np.abs(data))
    filtered_data = scipy.signal.wiener(data) 
    filtered_data = filtered_data - np.mean(filtered_data)
    filtered_data = filtered_data / np.max(np.abs(filtered_data))    
    new_name = os.path.basename(filename)
    save_path = output + '/' + new_name
    wave.write(save_path, sample_rate, filtered_data.astype(np.float32))
    # 这里重新读取一遍干净的音频，与denoised的音频文件对比
    clean_path = clean_path + '/' + new_name
    _, clean_data = wave.read(clean_path)
    clean_data = clean_data - np.mean(clean_data)
    clean_data = clean_data / np.max(np.abs(clean_data))
    # 设置一系列评价指标
    if(metrics == 'rmse'): # original:261.58, normalization:0.0235  all_norm:0.0132
        rmse = np.sqrt(mean_squared_error(clean_data, filtered_data))
        #print(f'均方根误差:{rmse}')
        return rmse
    elif(metrics == 'snr'): # -12.61  3.05  7.83
        noise = filtered_data - clean_data
        signal_power = np.sum(clean_data ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    elif(metrics == 'pesq'):
        from pesq import pesq
        try:
            pesq_score = pesq(sample_rate, clean_data[10000:], filtered_data[10000:], 'nb')
            return pesq_score
        except:
            return -999
    elif(metrics == 'stoi'): # 34.2% 34.2% 34.2%
        from pystoi import stoi
        stoi = stoi(data, filtered_data, sample_rate)
        return stoi
    elif(metrics == 'si_snr'): # 7.82
        eps = 1e-8
        noise = filtered_data - clean_data
        clean_data = clean_data - np.mean(clean_data)
        filtered_data = filtered_data - np.mean(filtered_data)
        scaling_factor = np.dot(clean_data, filtered_data) / (np.dot(clean_data, clean_data) + eps)
        projection = scaling_factor * clean_data
        noise = filtered_data - projection
        si_snr = 10 * np.log10(np.dot(projection, projection) / (np.dot(noise, noise) + eps))
        return si_snr
    if(visuralize!=0):
        # 在这里可视化一下滤波前后的波形
        sample_rate, data = wave.read(filename) 
        time = np.linspace(0, len(data) / sample_rate, num=len(data))
        plt.figure(figsize=(10, 4))
        plt.subplot(2,1,1)
        plt.plot(time, data)
        plt.title('mixed Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True) #显示网格
        #plt.show()
        plt.subplot(2,1,2)
        plt.plot(time, filtered_data)
        plt.title('filterd Signal',color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    path = 'SPRSound-Gaussian-Noise-main/tr/mix'
    output_path = 'SPRSound-Gaussian-Noise-main/tr/wiener_process'
    clean_path = 'SPRSound-Gaussian-Noise-main/tr/s1'
    wav_dir = os.listdir(path)
    metrics_list = []
    metrics = 0
    for i in range(len(wav_dir)):
        wav_path = path + '/' + wav_dir[i]
        metrics = wiener_filter(filename=wav_path, clean_path=clean_path, output=output_path, 
                                metrics='pesq')
        if(metrics != -999):
            metrics_list.append(float(metrics))
        print(f'第{i+1}个文件处理完成')
    mean_metrics = sum(metrics_list) / len(metrics_list)
    print(mean_metrics)


