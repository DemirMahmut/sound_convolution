import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import time

def myConv(x, n, y, m): # Convolution function
    result = [] # Result list
    for i in range(n+m-1): # Loop through the range of n+m-1
        sum = 0 # Initialize sum to 0
        for j in range(m): # Loop through the range of m
            if i-j >= 0 and i-j < n: # Check if i-j is in the range of x
                sum += x[i-j] * y[j] # Add the product of x[i-j] and y[j] to sum
        result.append(sum) # Append sum to result
        print(f"i: {i}, sum: {sum}") # Print i and sum
    return result # Return result

def findY(n, x, M): # Function to find y[n]
    result = np.zeros(n) # Initialize result to an array of zeros
    for i in range(n): # Loop through the range of n
        sum = 0 # Initialize sum to 0
        for j in range(1, M+1): # Loop through the range of 1 to M+1
            sum += 2**(-j) * j * x[n - 3000 * j] # Add the product of 2^(-j), j, and x[n - 3000 * j] to sum
        result[i] = x[i] + sum # Set result[i] to x[i] + sum
    return result # Return result

n = int(input("Enter the size of x[n]: "))
index_x = int(input("Enter the starting index of x[n]: "))
x = [] # Initialize x to an empty list
for i in range(n): # Loop through the range of n
    x.append(int(input(f"Enter the value of x[{i}]: "))) # Append the input to x
    
m = int(input("Enter the size of y[n]: ")) # Input the size of y[n]
index_y = int(input("Enter the starting index of y[n]: ")) # Input the starting index of y[n]
y = [] # Initialize y to an empty list
for i in range(m):
    y.append(int(input(f"Enter the value of y[{i}]: ")))

result = myConv(x, n, y, m) # My Convolution
result2 = np.convolve(x, y) # Numpy Convolution
print(f"x[n] = {x}")
print(f"y[n] = {y}")
print(f"Result myConv = {result}")
print(f"Result Numpy Convolution = {result2}")

plt.figure() # Plotting the graphs
plt.subplot(2, 2, 1)
plt.stem(range(index_x, index_x+n), x)
plt.title("x[n]")
plt.subplot(2, 2, 2)
plt.stem(range(index_y, index_y+m), y)
plt.title("y[n]")
plt.subplot(2, 2, 3)
plt.stem(range(index_x+index_y, index_x+index_y+len(result)), result)
plt.title("x[n] * y[n] (My Convolution)")
plt.subplot(2, 2, 4)
plt.stem(range(index_x+index_y, index_x+index_y+len(result2)), result2)
plt.title("x[n] * y[n] (Numpy Convolution)")  
plt.subplots_adjust(hspace= 1.0)
plt.show()

fs = 44100 # Sampling Frequency
print('Start speaking.')
recording = sd.rec(int(5 * fs), samplerate=fs, channels=1) # Recording
sd.wait() # Wait for the recording to finish
print('End of Recording.')
x1 = np.squeeze(recording) # Squeeze the recording

print('Start speaking.')
recording = sd.rec(int(10 * fs), samplerate=fs, channels=1) # Recording
sd.wait() # Wait for the recording to finish
print('End of Recording.')
x2 = np.squeeze(recording) # Squeeze the recording

y1_3 = findY(len(x1), x1, 3) 
y2_3 = findY(len(x2), x2, 3)

y1_4 = findY(len(x1), x1, 4)
y2_4 = findY(len(x2), x2, 4)

y1_5 = findY(len(x1), x1, 5)
y2_5 = findY(len(x2), x2, 5)

start = time.time()
numpyY1 = np.convolve(x1, y1_3) 
numpyY2 = np.convolve(x2, y2_3)
end = time.time()
sd.play(numpyY1, fs)
sd.wait()
print("Numpy Convolution Time: ", end - start)
sd.play(numpyY2, fs)
sd.wait()

start = time.time()
myY1 = myConv(x1, len(x1), y1_3, len(y1_3))
myY2 = myConv(x2, len(x2), y2_3, len(y2_3))
end = time.time()
print("My Convolution Time: ", end - start)
sd.play(myY1, fs)
sd.wait()
sd.play(myY2, fs)
sd.wait()

plt.figure()
plt.subplot(3, 2, 1)
plt.plot(range(len(x1)), x1)
plt.title("x1")
plt.subplot(3, 2, 2)
plt.plot(range(len(x2)), x2)
plt.title("x2")
plt.subplot(3, 2, 3)
plt.plot(range(len(y1_3)), y1_3)
plt.title("y1_3")
plt.subplot(3, 2, 4)
plt.plot(range(len(y2_3)), y2_3)
plt.title("y2_3")
plt.subplot(3, 2, 5)
plt.plot(range(len(numpyY1)), numpyY1)
plt.title("numpyY1")
plt.subplot(3, 2, 6)
plt.plot(range(len(numpyY2)), numpyY2)
plt.title("numpyY2")
plt.subplots_adjust(hspace= 1.0)
plt.show()

start = time.time()
numpyY1 = np.convolve(x1, y1_4)
numpyY2 = np.convolve(x2, y2_4)
end = time.time()
sd.play(numpyY1, fs)
sd.wait()
print("Numpy Convolution Time: ", end - start)
sd.play(numpyY2, fs)
sd.wait()

start = time.time()
myY1 = myConv(x1, len(x1), y1_4, len(y1_4))
myY2 = myConv(x2, len(x2), y2_4, len(y2_4))
end = time.time()
print("My Convolution Time: ", end - start)
sd.play(myY1, fs)
sd.wait()
sd.play(myY2, fs)
sd.wait()

plt.figure()
plt.subplot(3, 2, 1)
plt.plot(range(len(x1)), x1)
plt.title("x1")
plt.subplot(3, 2, 2)
plt.plot(range(len(x2)), x2)
plt.title("x2")
plt.subplot(3, 2, 3)
plt.plot(range(len(y1_4)), y1_4)
plt.title("y1_4")
plt.subplot(3, 2, 4)
plt.plot(range(len(y2_4)), y2_4)
plt.title("y2_4")
plt.subplot(3, 2, 5)
plt.plot(range(len(numpyY1)), numpyY1)
plt.title("numpyY1")
plt.subplot(3, 2, 6)
plt.plot(range(len(numpyY2)), numpyY2)
plt.title("numpyY2")
plt.subplots_adjust(hspace= 1.0)
plt.show()

start = time.time()
numpyY1 = np.convolve(x1, y1_5)
numpyY2 = np.convolve(x2, y2_5)
end = time.time()
sd.play(numpyY1, fs)
sd.wait()
print("Numpy Convolution Time: ", end - start)
sd.play(numpyY2, fs)
sd.wait()

start = time.time()
myY1 = myConv(x1, len(x1), y1_5, len(y1_5))
myY2 = myConv(x2, len(x2), y2_5, len(y2_5))
end = time.time()
print("My Convolution Time: ", end - start)
sd.play(myY1, fs)
sd.wait()
sd.play(myY2, fs)
sd.wait()

plt.figure()
plt.subplot(3, 2, 1)
plt.plot(range(len(x1)), x1)
plt.title("x1")
plt.subplot(3, 2, 2)
plt.plot(range(len(x2)), x2)
plt.title("x2")
plt.subplot(3, 2, 3)
plt.plot(range(len(y1_5)), y1_5)
plt.title("y1_5")
plt.subplot(3, 2, 4)
plt.plot(range(len(y2_5)), y2_5)
plt.title("y2_5")
plt.subplot(3, 2, 5)
plt.plot(range(len(numpyY1)), numpyY1)
plt.title("numpyY1")
plt.subplot(3, 2, 6)
plt.plot(range(len(numpyY2)), numpyY2)
plt.title("numpyY2")
plt.subplots_adjust(hspace= 1.0)
plt.show()