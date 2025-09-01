import math 
x1,x2=10,5
y1,y2=15,0 
distance=math.sqrt((x2-x1)**2+(y2-y1)**2)
print(distance) 
print(distance) 
'''def calculate_area(x1, y1, x2, y2, x3, y3):
    area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1)
    return area
# import matplotlib.pyplot as plt
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# plt.plot(x, y, 'r--', linewidth=2)
# plt.title('Customized Plot', fontsize=14)
# plt.xlabel('x-axis', fontsize=12)
# plt.ylabel('y-axis', fontsize=12)
# plt.grid(True)
# plt.show()
x1, y1 = 2, 3
x2,y2 = 5, 7
x3, y3 = 6, 1
triangle_area = calculate_area(x1, y1, x2, y2, x3, y3)
print(f"{triangle_area}") '''
# x = np.linspace(-2,2,100)
# # plt.plot(x, y, 'r--', linewidth=2, label='y = x^2',alpha=1,marker="*",markersize=1)
# # plt.title('Customized Plot', fontsize=14)
# # plt.xlabel('x-axis', fontsize=12)
# # plt.ylabel('y-axis', fontsize=12)
# # plt.legend()
# # plt.grid()  
# # plt.show()
# # x = np.linspace(0, 10, 100)
# # y = np.cos(x)
# # plt.plot(x, y, 'r--', linewidth=2)
# # plt.title('Customized Plot', fontsize=14)
# # plt.xlabel('x-axis', fontsize=12)
# # plt.ylabel('y-axis', fontsize=12)   
# # plt.grid(True)
# # plt.show()
# # # x = np.linspace(0, 10, 100)
# # # y = np.tan(x)
# # # plt.plot(x, y, 'r', linewidth=2)
# # # plt.title('Customized Plot',fontsize=14)
# # # plt.xlabel('x-axis',fontsize=12)
# # # plt.ylabel('y-axis',fontsize=12)
# # # plt.grid()
# # # plt.show()
# # x1=np.array([1,2,3,4,5])
# # y2=3*x1+3
# # plt.xlabel('x-axis')
# # plt.ylabel('y-axis')
# # plt.title('Graph')
# # plt.grid()
# # plt.plot(x1,y2)  
# # y = x
# # # plt.figure(figsize=(6, 6))
# # plt.plot(x, y, label='y = x', color='b',lw=2,ls='-',marker='*',markersize=2)
# # plt.xlabel("X-axis")
# # plt.ylabel("Y-axis")
# # plt.title("Graph of y = x")
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# # days = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # stock_prices = np.array([100, 105, 102, 108, 110, 115, 112, 118, 120, 122])

# # plt.plot(days, stock_prices, marker='o', color='blue', linestyle='-', linewidth=2)
# # plt.fill_between(days, stock_prices, color="skyblue", alpha=0.3)
# # for i, price in enumerate(stock_prices):
# #     plt.text(days[i], price + 1, str(price), ha='center')
# # plt.title("স্টক মার্কেট ডেটা")
# # plt.xlabel("দিন")
# # plt.ylabel("স্টকের দাম (টাকা)")
# # plt.grid(True)
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# x3 = np.linspace(0,10,100)
# y = 3*x3 + 6
# y1 = 3*x3 + 3
# y2 = np.full_like(x3, 2)  # এই লাইনটি যোগ করা হয়েছে

# plt.plot(x3, y, label='y = 3x+6', color='b', lw=2, ls='-', marker='*', markersize=2)
# plt.plot(x3, y1, label='y = 3x+3', color='r', lw=2, ls='-', marker='*', markersize=2)
# plt.plot(x3, y2, label='y = 2', color='g', lw=2, ls='-', marker='*', markersize=2)

# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('Graph')
# plt.legend()
# plt.grid()
# plt.show()



x1,x2=2,3 
y1,y2 = 5,7
x3,y3 = 6,1
area=abs(x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 + y2 * x3 + y3 * x1)
print(area)
x1,x2=2,3 
y1,y2 = 5,7
slope=(y2-y1)/(x2-x1)
print(slope)
import numpy as np
print(type(np.nan))
arrays=np.zeros((2,3),str)
x=np.array([1,2,3,4,5,6,7,8,9])
print(type(x))
a1=np.array([[1,2,3],[4,5,6],[7,1,0]])
print(a1)
z=a1.sum(axis=0)
print(z)
z=a1.sum(axis=1)
print(z)
print(a1.argmin())
print(a1.argmax())
num=np.array([1,2,3,4,5,6,7,8,9])
print(np.split(num,3))
print(num)
arrays=np.ones((2,3),bool)
print(arrays)
arrays=np.ones((2,3),int)
print(arrays)
x=np.array([2,4,6,8])
print(x)
print(x.shape)
y=np.arange(1,15,2)
print(y)
a=np.ones((2,5))
print(a)
z=a.flatten()
print(z)
print(z.transpose())
x1=np.array([[1,0,5],[3,4,5]])
print(x1)
print(x1.nonzero())
print(x.dtype)
print(x.nbytes)
num=np.array([1,2,3,4,5,6,7,8,9])
print(np.split(num,3))
print(num)
x = np.linspace(0, 10, 100)
d_array= np.random.random(5)
print(d_array)
# x=[1,2,3,4,5]
# x=np.array(x)

d_array= np.random.randint(5)
print(d_array)
_array= np.eye(5)
print(_array)
d_= np.arange(1,10,2)
print(d_)
numpy_array= np.array([1,3,4,2,6,7,9,8,5])
print(numpy_array)
sorted_array= np.sort(numpy_array)
print(sorted_array)
matrix=np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
print(matrix)
print(matrix[0,0])
print(matrix[1,1])
matrix1=np.array([[10,12,13],
                  [14,15,16],
                  [17,18,19]])
print(matrix1)
print(np.dot(matrix,matrix1))
print(np.linalg.det(matrix))
print(np.cross(matrix,matrix1))
list=[[2,4,6,8,10],
     [12,14,16,18,20],
     [22,24,26,28,30],
     [32,34,36,38,40],
     [42,44,46,48,50]]
print(list)
print(list[0][1])
print(np.dot(matrix,matrix1))
print(np.cross(matrix,matrix1))

matrix = np.array([[4, 7],
                   [2, 6]])

print(np.linalg.det(matrix))     
print("After Determinant existed:",np.linalg.inv(matrix))     

vector1 = np.array([2,3,1])
vector2 = np.array([1,4,2])
dot_product = np.dot(vector1, vector2)
print("Dot product:", dot_product)
cross_product = np.cross(vector1, vector2)
print("Cross product:", cross_product)
print("Magnitude of vector1:", np.linalg.norm(vector1))
print("Magnitude of vector2:", np.linalg.norm(vector2))

after_dot = dot_product / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
angle_degrees = np.degrees(np.arccos(after_dot))
print("Angle between vector1 and vector2 in degrees:", angle_degrees)
after_cross=cross_product / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
angle_in_degrees = np.degrees(np.arcsin(after_cross))
import matplotlib.pyplot as plt
import numpy as np

# Define the x range, avoiding points near the asymptotes at odd multiples of pi/2
x = np.linspace(-1 * np.pi, 1 * np.pi, 1000)
y = np.sin(x)

# Identify and replace points where the function jumps from +inf to -inf with NaN to prevent connecting lines
y[:-1][np.diff(y) < 0] = np.nan

# Plot the function with axis limits to focus on the main behavior
plt.plot(x, y)
plt.ylim(-10, 10)
plt.xlim(-1 * np.pi, 1 * np.pi)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("$\\tan(x)$")
plt.title("$y = \\sin(x)$", fontsize=14)

# Set x-axis labels to show key angles in radians
radian_multiples = [-2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2]
radians = [n * np.pi for n in radian_multiples]
radian_labels = ['$-2\\pi$', '$-3\\pi/2$', '$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$']
plt.xticks(radians, radian_labels)

plt.show()   

