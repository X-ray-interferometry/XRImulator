import matplotlib.pyplot as plt
import numpy as np

zero = np.array([0, 0])
theta = np.array([1, 1/2])
zerototheta = np.array([[0,1], [0, 1/2]])

fig = plt.figure(figsize=(6,6))
plt.plot([-4, 4], [0, 0], 'b-')
plt.plot(zerototheta[0,:], zerototheta[1,:], 'r')
plt.plot(theta[0], theta[1], 'ro')
plt.plot([0, theta[0]], [0, 0], 'g')
plt.plot([theta[0], theta[0]], [0, theta[1]], 'k--')
plt.plot(theta[0], 0, 'go')
plt.xlabel('$\phi$ ')
plt.ylabel('$\psi$ ')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()

sat_axis = np.array([[-4, 4], [-4, 4]])
meet_point = [3/4, 3/4]
arc_rad = np.linspace(0, np.pi/4, 100)
arc_coord = np.stack((np.array([.5*np.cos(i) for i in arc_rad]), np.array([.5*np.sin(i) for i in arc_rad])), axis=-1)

fig = plt.figure(figsize=(6,6))

plt.plot(sat_axis[0,:], sat_axis[1,:], 'b')
plt.plot(zerototheta[0,:], zerototheta[1,:], 'r')
plt.plot(theta[0], theta[1], 'ro')
plt.plot([0, meet_point[0]], [0, meet_point[1]], 'g')
plt.plot(meet_point[0], meet_point[1], 'go')
plt.plot([theta[0], meet_point[0]], [theta[1], meet_point[1]], 'k--')
plt.plot(arc_coord[:,0], arc_coord[:,1], '--', color='gray')
plt.xlabel('$\phi$')
plt.ylabel('$\psi$')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()