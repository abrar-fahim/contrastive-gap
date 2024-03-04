
# visualize n points on unit 3D sphere

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



n = 100


# points along diameter circle
theta = np.linspace(0, 2*np.pi, n)
xd = np.cos(theta)
yd = np.zeros(n)
zd = np.sin(theta)


# normalize
magnitude = np.sqrt(xd**2 + yd**2 + zd**2)
xd /= magnitude
yd /= magnitude
zd /= magnitude



# generate n points on a ring
theta = np.linspace(0, 2*np.pi, n)
x = np.cos(theta)
y = np.sin(theta)
z = np.zeros(n) - 0.5






# normalize
magnitude = np.sqrt(x**2 + y**2 + z**2)
x /= magnitude
y /= magnitude
z /= magnitude

# keep only n/3 points
x = x[:n//3]
y = y[:n//3]
z = z[:n//3]


# generate another ring rotated 
theta2 = np.linspace(0, 2*np.pi, n)
x2 = np.cos(theta2) 
y2 = np.sin(theta2) 
# z2 = np.cos(theta2) + 3
z2 = np.cos(theta2) + np.sin(theta2) - 1    


# normalize to unit sphere
magnitude = np.sqrt(x2**2 + y2**2 + z2**2)
x2 /= magnitude
y2 /= magnitude
z2 /= magnitude

# keep only n/3 points
x2 = x2[:n//3]
y2 = y2[:n//3]
z2 = z2[:n//3]

# 3rd ring rotated
theta3 = np.linspace(0, 2*np.pi, n)
x3 = np.cos(theta3)
y3 = np.sin(theta3)
z3 = np.zeros(n) + 0.75

#normalize to unit sphere
magnitude = np.sqrt(x3**2 + y3**2 + z3**2)
x3 /= magnitude
y3 /= magnitude
z3 /= magnitude

# keep only n/3 points
x3 = x3[:n//3]
y3 = y3[:n//3]
z3 = z3[:n//3]






# find centroid
x_centroid = np.mean(x)
y_centroid = np.mean(y)
z_centroid = np.mean(z)

# find centroid 2
x_centroid2 = np.mean(x2)
y_centroid2 = np.mean(y2)
z_centroid2 = np.mean(z2)

# find centroid 3
x_centroid3 = np.mean(x3)
y_centroid3 = np.mean(y3)
z_centroid3 = np.mean(z3)

n = len(x)



# x is reference

# find mean pairwise cosine similarity for matching points 
cosine_similarities = []
for i in range(n):
    cosine_similarities.append(np.dot([x[i], y[i], z[i]], [x2[i], y2[i], z2[i]]))

mean_cosine_similarity12 = np.mean(cosine_similarities)

print(f'mean_cosine_similarity between x (RED) and x2 (GREEN): {mean_cosine_similarity12}')

# find mean pairwise cosine similarity for matching points
cosine_similarities = []
for i in range(n):
    cosine_similarities.append(np.dot([x[i], y[i], z[i]], [x3[i], y3[i], z3[i]]))

mean_cosine_similarity13 = np.mean(cosine_similarities)

print(f'mean_cosine_similarity between x (RED) and x3 (BLUE): {mean_cosine_similarity13}')

# eucldiean distance between centroids
euclidean_distance12 = np.linalg.norm(np.array([x_centroid, y_centroid, z_centroid]) - np.array([x_centroid2, y_centroid2, z_centroid2]))

print(f'euclidean_distance between centroids of x (RED) and x2 (GREEN): {euclidean_distance12}')

# eucldiean distance between centroids
euclidean_distance13 = np.linalg.norm(np.array([x_centroid, y_centroid, z_centroid]) - np.array([x_centroid3, y_centroid3, z_centroid3]))

print(f'euclidean_distance between centroids of x (RED) and x3 (BLUE): {euclidean_distance13}')

# for each point, matching point is the one with the highest cosine similarity
# find the matching point for each point
matching_points12 = []
for i in range(n):
    cosine_similarities = []
    for j in range(n):
        cosine_similarities.append(np.dot([x[i], y[i], z[i]], [x2[j], y2[j], z2[j]]))

    if cosine_similarities[i] == max(cosine_similarities):
        matching_points12.append(i)
    else:
        # print(f'cosine_similarities12 wrong for {i}:', cosine_similarities)
        matching_points12.append(np.argmax(cosine_similarities))

# check how many points are matched to the correct point
correct_matches = 0
for i in range(n):
    if matching_points12[i] == i:
        correct_matches += 1


print('matching_points12:', matching_points12)
print(f'correct_matches between x (RED) and x2 (GREEN): {correct_matches / n}')

# for each point, matching point is the one with the highest cosine similarity
# find the matching point for each point
matching_points13 = []
for i in range(n):
    cosine_similarities = []
    for j in range(n):
        cosine_similarities.append(np.dot([x[i], y[i], z[i]], [x3[j], y3[j], z3[j]]))

    if cosine_similarities[i] == max(cosine_similarities):
        matching_points13.append(i)
    else:
        matching_points13.append(np.argmax(cosine_similarities))

    # if i != matching_points13[i]:
        # print(f'cosine_similarities13 for i={i}:', cosine_similarities)


# print('cosine_similarities13:', cosine_similarities)

# check how many points are matched to the correct point
correct_matches = 0
for i in range(n):
    if matching_points13[i] == i:
        correct_matches += 1

print('matching_points13:', matching_points13)

print(f'correct_matches between x (RED) and x3 (BLUE): {correct_matches / n}')


    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xd, yd, zd, c='k')
ax.scatter(x, y, z, c='r')
ax.scatter(x_centroid, y_centroid, z_centroid, c='r')
ax.scatter(x2, y2, z2, c='g')
ax.scatter(x_centroid2, y_centroid2, z_centroid2, c='g')
ax.scatter(x3, y3, z3, c='b')
ax.scatter(x_centroid3, y_centroid3, z_centroid3, c='b')

# draw lines between matching points between x and x3
# for i in range(n):
#     a = [x[i], x2[matching_points12[i]]]
#     b = [y[i], y2[matching_points12[i]]]
#     c = [z[i], z2[matching_points12[i]]]
#     ax.plot(a,b,c, c='grey', alpha=0.5)

# set axes to be equal
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_box_aspect([1,1,1])

plt.show()



