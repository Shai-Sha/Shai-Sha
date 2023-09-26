import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Harrison's one-line solution!
x1, x2, y, z = np.loadtxt('assign1_data.txt', skiprows = 2, unpack = True)
        
# Part 1 --------------------------------------

def solve1(x, y):
    m = np.sum((x-np.mean(x)) * (y-np.mean(y))) / np.sum((x-np.mean(x))**2)
    b = np.mean(y) - m * np.mean(x)
    return m, b

m1, b1 = solve1(x1, y)
print('Part 1, x1: m = ' + str(m1) + '   b  = ' + str(b1))
m2, b2 = solve1(x2, y)
print('Part 1, x2: m = ' + str(m2) + '   b  = ' + str(b2))

# Extra
xx = np.array([-.2, 1.2])
yy = m1 * xx + b1
plt.scatter(x1, y)
plt.plot(xx, yy, 'r')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Part 1')
plt.show()

# Part 2 --------------------------------------

def solve(x1, x2, y):
    A = np.vstack([x1, x2, np.ones(len(x1))]).T
    w1,w2,b = np.linalg.lstsq(A, y, rcond=None)[0]
    return w1,w2,b

w1, w2, b = solve(x1, x2, y)
print('\nPart 2, w1 = ' + str(w1) + '   w2 = ' + str(w2) + '   b  = ' + str(b))

# Extra, 2D
for k in range(len(x1)):
    plt.plot(x1[k], y[k],'b*' if z[k] else 'r*')
    plt.plot(x2[k], y[k], 'bo' if z[k] else 'ro')
xx = np.array([-.2, 1.2])
yy = w1 * xx + w2* xx + b
plt.plot(xx, yy, 'k')
plt.title('Part 2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Extra, 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plt.hold(True)
for i in range(len(z)):
    if z[i] == 0:
        # We want red dots if they are classified as 0 (False)
        ax.scatter(x1[i], x2[i], y[i], c='r', marker='o')
    else:
        # We want blue dots if they are classified as 1 (True)
        ax.scatter(x1[i], x2[i], y[i], c='b', marker='o')
X = np.arange(0,1.2,(1/12))
Y = np.arange(0,1.2,(1/12))
X,Y = np.meshgrid(X,Y)

# Choose regression or classification -----------

#plt.title("3D Regression Model")
#zs = np.array([w1*X[i]+w2*Y[i]+b for i in range(len(X))])

plt.title("3D Classifier Model")
zs = np.zeros(X.shape)

# ------------------------------------------------

Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, color='gray', linewidth=0, alpha = 0.7, shade=False)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()


# Part 3 --------------------------------------

def test(label, w1, x1, w2, x2, b, z):

    # Run linear model to predict Y
    y = w1*x1 + w2*x2 + b

    # Turn Y into True/False
    y = y > 0

    # Compare Y to Z, getting True/False array
    success = (y == z)

    # Sum up individual successes to get overall success
    success = np.sum(success)

    # Total number of examples is length of Z (or of X1 or X2 or Y)
    n = len(z)

    # Report fraction success as percent
    print(label + ': ' + str(100 * success/n) + '% correct')

test('\nPart3', w1, x1, w2, x2, b, z)

# Part 4 --------------------------------------

print('\nPart 4:')

# Training/testing pairs
for t in [25,50,75]:
    w1, w2, b = solve(x1[:t], x2[:t], y[:t])
    test('    ' + str(t) + ': ', w1, x1[t:], w2, x2[t:], b, z[t:])

# "Zero" model
w1,w2, b = 0,0,0
test('with zero model', w1, x1, w2, x2, b, z)
    








