import matplotlib.pyplot as plt
import numpy as np

"""# Create some example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 5, 7, 7, 10])"""
print("Hello Simon")
x1=[]
x2=[]
y=[]

with open('assign1_data.txt') as f:
	next(f) # skip 1 line
	next(f) # skip 1 line
	for line in f:
		data=line.split()
		x1.append(float(data[0]))
		x2.append(float(data[1]))
		y.append(float(data[2]))
        
"""for i in range(len(x1)):
    print(x1[i],y[i])"""

# Plot the array of dots
plt.scatter(x1, y, color='red', label='Dots')


# Fit a linear regression line
slope, intercept = np.polyfit(x1, y, 1)
linear_fit = np.array(slope) * x1 + intercept
print(f"y={slope} * x1 +{intercept}")
slope, intercept = np.polyfit(x2, y, 1)
print(f"y={slope} * x2 +{intercept}")

# Plot the linear graph
plt.plot(x1, linear_fit, color='blue', label='Linear Fit')

# Add labels and legend
plt.xlabel('X1-axis')
plt.ylabel('Y-axis')
plt.title('Array of Dots and Linear Graph')
plt.legend()

# Display the plot
plt.show()
